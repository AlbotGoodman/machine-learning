import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class Preprocessing:
    """Handles preprocessing for the content-based filtering model."""


    def __init__(self):
        nltk.download("stopwords")


    def genre_cleanup(self, df):
        """
        Cleans the genres column by dropping movies without genres and formats the text.
        Formatting includes:
        - lowercasing
        - replacing symbol separators with blank spaces
        - removing hyphens
        - removing leading and trailing whitespaces
        
        Arguments:
        df -- DataFrame with movie_id, title and genres columns

        Returns:
        df -- DataFrame with movie_id and genres columns
        """

        df["genres"] = df["genres"].replace("(no genres listed)", np.nan)
        df["genres"] = df["genres"].str.lower().str.replace("|", " ").str.replace("-", "").str.strip()
        df = df.dropna(how="any")
        df = df.drop(columns=["title"])
        
        return df
    

    def _text_formatting(self, text):
        """
        Helper method to format text in a string.
        Formatting includes:
        - lowercasing
        - remove text in square brackets
        - remove links
        - remove HTML tags
        - remove punctuation
        - remove newline characters
        - remove words containing numbers

        Arguments:
        text -- String to be formatted

        Returns:
        text -- Formatted string
        """

        text = str(text).lower()
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>+", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\n", "", text)
        text = re.sub(r"\w*\d\w*", "", text)
        return text
    

    def _remove_stopwords(self, text):
        """
        Helper method to remove stopwords from a string.

        Arguments:
        text -- String to remove stopwords from

        Returns:
        text -- String without stopwords
        """
        
        stop_words = stopwords.words("english")
        return " ".join(word for word in text.split(" ") if word not in stop_words)
    

    def _stem_text(self, text):
        """
        Helper method to stem words in a string.

        Arguments:
        text -- String to stem words in

        Returns:
        text -- String with stemmed words
        """
        
        stemmer = nltk.SnowballStemmer("english")
        return " ".join(stemmer.stem(word) for word in text.split(" "))


    def tag_cleanup(self, df):

        df = df.drop(columns=["user_id", "timestamp"])
        df = df.groupby("movie_id")["tag"].apply(lambda x: " ".join(x.astype(str))).reset_index()

        df["tag"] = df["tag"].apply(self._text_formatting)
        df["tag"] = df["tag"].apply(self._remove_stopwords)
        df["tag"] = df["tag"].apply(self._stem_text)

        return df
    

    def combine_dataframes(self, df_genres, df_tags):
        """
        Combines the genres and tags DataFrames into a single DataFrame.

        Arguments:
        df_genres -- The cleaned genres DataFrame
        df_tags -- The cleaned tags DataFrame

        Returns:
        df -- Combined DataFrame with movie_id and semantics columns
        """

        df = df_genres.merge(df_tags, on="movie_id", how="left")
        df["semantics"] = df.apply(lambda x: f"{x["genres"]} {x["tag"]}" if pd.notna(x["tag"]) else x["genres"], axis=1)
        df = df.drop(columns=["genres", "tag"])

        return df
    

    def pipeline_processing(self, df_genres, df_tags):
        """
        A pipeline method for preprocessing the movies DataFrame. 

        Arguments:
        df_genres -- DataFrame with movie_id, title and genres columns
        df_tags -- DataFrame with user_id, movie_id, tag and timestamp columns

        Returns:
        df -- DataFrame with movie_id and semantics columns
        """

        df_genres = self.genre_cleanup(df_genres)
        df_tags = self.tag_cleanup(df_tags)
        df = self.combine_dataframes(df_genres, df_tags)

        return df
    

class Modelling:
    """Handles the modelling for the content-based filtering model."""


    def __init__(self):
        self.tfidf_matrix = None
        self.lsa_matrix = None


    def create_tfidf_matrix(self, df):
        """
        Creates a TF-IDF matrix from the semantics column of the DataFrame.

        Arguments:
        df -- DataFrame with movie_id and semantics columns

        Returns:
        self -- updates instance variables
        """

        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(df["semantics"])
        self.tfidf_vocab = self.tfidf.get_feature_names_out()

        return self
    

    def create_lsa_model(self):
        """
        Creates a Latent Semantic Analysis (LSA) model from the TF-IDF matrix.

        Arguments:
        self -- uses instance variables

        Returns:
        self -- updates instance variables
        """

        lsa = TruncatedSVD(n_components=950)
        self.lsa_matrix = lsa.fit_transform(self.tfidf_matrix)

        return self
    

class Recommending:
    """Generates recommendations based on the trained content-based filtering model."""


    def __init__(self, model, movies_df):
        self.model = model
        self.df = movies_df
        self.H = model.lsa_matrix
        self.movie_mapper = None
        self.movie_mapper_reverse = None

    
    def _create_mapper(self):
        """
        Helper method that creates a mapper for movie_id to index and vice versa.

        Arguments:
        df -- DataFrame with movie_id, title and genres columns

        Returns:
        self -- updates instance variables
        """

        self.movie_mapper = {movie_id: idx for idx, movie_id in enumerate(self.df["movie_id"])}
        self.movie_mapper_reverse = {idx: movie_id for idx, movie_id in enumerate(self.df["movie_id"])}

        return self
    

    def get_recommendations(self, input_movies, n_recs=5):
        """
        Provides movie recommendations based on the input movie titles.
        
        Arguments:
        input -- List of movie IDs
        n_recs -- Number of recommendations to return

        Returns:
        recommendations -- DataFrame with movie IDs, titles and similarity scores
        """

        self._create_mapper()
        input_indices = [self.movie_mapper[movie_id] for movie_id in input_movies]
        input_matrix = self.H[input_indices, :]
        similarity_scores = cosine_similarity(input_matrix, self.H)

        # Find the highest similarity score for each candidate movie
        similarity_scores = np.max(similarity_scores, axis=0)
        similarity_scores[input_indices] = 0

        # Sort the scores without affecting the order
        sorted_scores = np.sort(similarity_scores)[::-1]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_ids = [self.movie_mapper_reverse[idx] for idx in sorted_indices]

        # Create a DataFrame with the top recommendations
        movie_titles = [title for movie_id in sorted_ids[:n_recs] for title in self.df[self.df["movie_id"] == movie_id]["title"].values]
        top_movies = [(movie, title, score) for movie, title, score in zip(sorted_ids, movie_titles, sorted_scores)][:n_recs]
        top_df = pd.DataFrame(top_movies, columns=["movie_id", "title", "score"])
        top_df = top_df.sort_values("score", ascending=False)

        return top_df
    

def main():

    # Load data
    movies = pd.read_csv("../data/movielens/movies.csv")
    tags = pd.read_csv("../data/movielens/tags.csv")

    # Preprocess data
    preprocessor = Preprocessing()
    semantics = preprocessor.pipeline_processing(movies, tags)

    # Create model
    content_model = Modelling()
    content_model.create_tfidf_matrix(semantics)
    content_model.create_lsa_model()

    # Provide recommendations
    content_rec = Recommending(content_model, movies)
    recommendations = content_rec.get_recommendations([58559], 10)
    print(recommendations)


if __name__ == "__main__":
    main()