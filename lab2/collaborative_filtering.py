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
    

    def pipeline(self, df):
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
        self.movie_mapper = None
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
        self.movie_mapper = {movie: i for i, movie in enumerate(unique_movies)}
        self.movie_mapper_reverse = {i: movie for i, movie in enumerate(unique_movies)}

        # Map original IDs to indices (will need to be reversed later)
        rows = np.array([user_mapper[user] for user in df["user_id"]])
        cols = np.array([self.movie_mapper[movie] for movie in df["movie_id"]])
        vals = df["rating"].values
        self.user_movie_matrix = csr_matrix((vals, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

        return self


    def create_nmf_model(self):
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
        self.H = model.H
        self.movie_mapper = model.movie_mapper
        self.movie_mapper_reverse = model.movie_mapper_reverse
        self.movies = movies_df


    def get_recommendations(self, input_movies, n_recs=5):
        """
        Provides movie recommendations based on the input movie titles.
        
        Arguments:
        input -- List of movie IDs
        n_recs -- Number of recommendations to return

        Returns:
        recommendations -- DataFrame with movie IDs, titles and similarity scores
        """
        input_indices = [self.movie_mapper[movie] for movie in input_movies]
        input_matrix = self.H[:, input_indices].transpose()
        similarity_scores = cosine_similarity(input_matrix, self.H.T)

        # Find the highest similarity score for each candidate movie
        similarity_scores = np.max(similarity_scores, axis=0)
        similarity_scores[input_indices] = 0

        # Sort the scores without affecting the order
        sorted_scores = np.sort(similarity_scores)[::-1]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_movies = [self.movie_mapper_reverse[idx] for idx in sorted_indices]

        # Translate IDs to titles and create a DataFrame
        movie_titles = [title for id in sorted_movies[:n_recs] for title in self.movies[self.movies["movie_id"] == id]["title"].values]
        top_movies = [(movie, title, score) for movie, title, score in zip(sorted_movies, movie_titles, sorted_scores)][:n_recs]
        top_df = pd.DataFrame(top_movies, columns=["movie_id", "title", "score"])
        top_df = top_df.sort_values("score", ascending=False)

        return top_df
    

def main():

    # Load data
    ratings = pd.read_csv("../data/movielens/ratings.csv", usecols=["user_id", "movie_id", "rating"])
    movies = pd.read_csv("../data/movielens/movies.csv")
    
    # Preprocess data
    preprocessor = Preprocessing()
    ratings = preprocessor.pipeline(ratings)
    movies = Preprocessing.get_matching_movies(ratings, movies)
    joblib.dump(ratings, "joblib/collab/ratings_processed.joblib")
    joblib.dump(movies, "joblib/collab/movies_processed.joblib")
    
    # Create model
    collab_model = Modelling()
    collab_model.create_user_movie_matrix(ratings)
    collab_model.create_nmf_model()
    joblib.dump(collab_model, "joblib/collab/nmf_150.joblib")
    
    # Create recommendations
    collab_rec = Recommending(collab_model, movies)
    recommendations = collab_rec.get_recommendations([152081, 134853, 6377], 10)
    print(recommendations)


if __name__ == "__main__":
    main()