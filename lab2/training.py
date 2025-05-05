import pandas as pd
import numpy as np
import joblib
from collaborative_filtering import (Preprocessing as CFPreprocessing, 
                                    Modelling as CFModelling, 
                                    Recommending as CFRecommending)
from content_based_filtering import (Preprocessing as CBFPreprocessing,
                                    Modelling as CBFModelling,
                                    Recommending as CBFRecommending)

# Load data
ratings = pd.read_csv("../data/movielens/ratings.csv", usecols=["user_id", "movie_id", "rating"])
movies = pd.read_csv("../data/movielens/movies.csv")
tags = pd.read_csv("../data/movielens/tags.csv")

# Collaborative filtering
cf_preprocessor = CFPreprocessing()
ratings_processed = cf_preprocessor.pipeline(ratings)
movies_processed = CFPreprocessing.get_matching_movies(ratings_processed, movies)

cf_model = CFModelling()
cf_model.create_user_movie_matrix(ratings_processed)
cf_model.create_nmf_model()

# Content-based filtering
cbf_preprocessor = CBFPreprocessing()
semantics = cbf_preprocessor.pipeline_processing(movies, tags)

cbf_model = CBFModelling()
cbf_model.create_tfidf_matrix(semantics)
cbf_model.create_lsa_model()

# Save variables
joblib.dump(cf_model, "joblib/collab/nmf_150.joblib")
joblib.dump(semantics, "joblib/content/semantics.joblib")
joblib.dump(cbf_model, "joblib/content/lsa_950.joblib")