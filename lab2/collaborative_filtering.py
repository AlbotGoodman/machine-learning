import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import MiniBatchNMF
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class Preprocessing:
    """Handles preprocessing for the collaborative filtering model."""


    def __init__(self):
        pass

    
    @staticmethod
    def filtering(df, col):
        """
        Using distribution quantiles to filter out the lower end of the data for statistical significance.
        Also sampling a capped number of ratings from movies/users with a high number of ratings to reduce bias.
        
        Arguments:
        df -- DataFrame with user_id, movie_id and rating columns
        col -- Column to filter on, either "user_id" or "movie_id"

        Returns:
        df -- DataFrame filtered by the specified column
        """

        lower_quantile = 0.50 if col == "user_id" else 0.75
        higher_quantile = 0.90

        # Filtering out {col} with few rating contributions
        group_ratings = df.groupby(col).size()
        filtered_group_list = group_ratings[group_ratings >= group_ratings.quantile(lower_quantile)].index
        filtered_ratings = df[df[col].isin(filtered_group_list)]

        # Calculate the number of ratings per {col}, then drop the {col} with fewer than cap and create a list of {col}_ids
        cap = int(group_ratings.quantile(higher_quantile))
        group_counts = filtered_ratings.groupby(col)["rating"].size()
        outliers_list = group_counts[group_counts > cap].index.tolist()

        # Iterate over each {col} and sample cap number of ratings
        outliers_collection = []
        outliers_df = filtered_ratings[filtered_ratings[col].isin(outliers_list)]
        for _, col_df in outliers_df.groupby(col):
            random_indices = np.random.choice(col_df.shape[0], cap, replace=False)
            col_df_capped = col_df.iloc[random_indices]
            outliers_collection.append(col_df_capped)
        outliers_capped = pd.concat(outliers_collection)

        # Remove whale data and insert the sampled whale data
        filtered_ratings = filtered_ratings[filtered_ratings[col].isin(outliers_list) == False]
        filtered_ratings = pd.concat([filtered_ratings, outliers_capped], ignore_index=True)

        return filtered_ratings
    

    @staticmethod
    def scaling(df):
        """
        Scales (standardising and normalising) the ratings of each user. 
        Standardising is necessary since the ratings overall are not normally distributed 
        and some users might always rate high, low or with a large variance. 
        Normalising is necessary since matrix factorisation can only handle non-negative values.

        Arguments:
        df -- DataFrame with user_id, movie_id and rating columns

        Returns:
        df -- DataFrame with user_id, movie_id and scaled rating columns
        """
        df_collection = []
        for user, user_df in df.groupby("user_id"): 
            rating_values = user_df["rating"].values
            mean = np.mean(rating_values)
            std = np.std(rating_values)
            if std < 1e-10:           # If there's no deviation in a users ratings, 
                user_df["rating"]=0.5 # set rating to normalised average to avoid division by zero.
            else:
                std_values = (rating_values - mean) / std
                min_std = np.min(std_values)
                max_std = np.max(std_values)
                user_df["rating"] = (std_values - min_std) / (max_std - min_std) # normalising
            df_collection.append(user_df)
        return pd.concat(df_collection)
    

    def filter_scaler(self, df):
        """
        A pipeline for preprocessing the ratings DataFrame.

        Arguments:
        df -- ratings DataFrame

        Returns:
        df -- returns a filtered and scaled ratings DataFrame
        """
        df = self.filtering(df, "user_id")
        df = self.filtering(df, "movie_id")
        df = self.scaling(df)
        return df
    

    def get_matching_movies(rating_df, movie_df):
        """
        Makes sure that both DataFrames contain the same movies. 

        Arguments:
        rating_df -- Preprocessed DataFrame containing fewer movies than the original dataset
        movie_df -- Original DataFrame containing all movies with titles and years

        Returns:
        movie_df -- DataFrame of movies that are in rating_df
        """
        return movie_df[movie_df["movie_id"].isin(rating_df["movie_id"].unique())]


class Modelling:
    """Uses users, movies and ratings to create a matrix facorisation model."""


    def __init__(self):
        self.user_mapper_reverse = None
        self.movie_mapper_reverse = None
        self.user_movie_matrix = None
        self.model = None
        self.W = None
        self.H = None


    def create_user_movie_matrix(self, df):
        """
        Creates a sparse matrix for matrix factorisation.

        Arguments:
        df -- DataFrame with user_id, movie_id and rating columns

        Returns: 
        self -- updates instance variables
        """
        
        unique_users = np.sort(df["user_id"].unique())
        unique_movies = np.sort(df["movie_id"].unique())

        # Creates a dictionary with an index for each user and movie - or reversed
        user_mapper = {user: i for i, user in enumerate(unique_users)}
        movie_mapper = {movie: i for i, movie in enumerate(unique_movies)}
        self.user_mapper_reverse = {i: user for i, user in enumerate(unique_users)}
        self.movie_mapper_reverse = {i: movie for i, movie in enumerate(unique_movies)}

        # Map original IDs to indices (will need to be reversed later)
        rows = np.array([user_mapper[user] for user in df["user_id"]])
        cols = np.array([movie_mapper[movie] for movie in df["movie_id"]])
        vals = df["rating"].values
        self.user_movie_matrix = csr_matrix((vals, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

        return self


    def train_model(self):
        """
        Trains a model based on MiniBatchNMF using set parameters found during explorative data analysis.

        Arguments:
        self -- uses only the instance variable for user-movie matrix

        Returns:
        self -- updates instance variables
        """
        self.model = MiniBatchNMF(n_components=150, batch_size=5000, alpha_W=0.001, alpha_H=0.01, l1_ratio=0.65)
        self.W = self.model.fit_transform(self.user_movie_matrix)
        self.H = self.model.components_
        return self


class Recommending:
    """Generates recommendations based on the trained collaborative filtering model."""


    def __init__(self, model, movies_df):
        self.model = model
        self.user_movie_matrix = model.user_movie_matrix
        self.W = model.W
        self.H = model.H
        self.user_mapper_reverse = model.user_mapper_reverse
        self.movie_mapper_reverse = model.movie_mapper_reverse
        self.movies = movies_df


    def get_recommendations(self, input, n_recommendations=5):
        """
        Provides movie recommendations based on the input movie titles.
        
        Arguments:
        input -- List of movie IDs
        n_recommendations -- Number of recommendations to return

        Returns:
        recommendations -- DataFrame with recommended movie titles
        """
