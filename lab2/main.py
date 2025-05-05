# DEPENDENCIES

import streamlit as st
from collaborative_filtering import Recommending as CFRecommending
from content_based_filtering import Recommending as CBFRecommending
import pandas as pd
import numpy as np
import joblib
st.set_page_config(page_title="AsMR")


# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("../data/movielens/movies.csv")


movies = load_data()


# COLLABORATIVE FILTERING

@st.cache_resource
def load_collab():
    return joblib.load("joblib/collab/nmf_150.joblib")


cf_model = load_collab()
cf_recommender = CFRecommending(cf_model, movies)


# CONTENT-BASED FILTERING

@st.cache_resource
def load_content():
    return joblib.load("joblib/content/lsa_950.joblib")


cbf_model = load_content()
cbf_recommender = CBFRecommending(cbf_model, movies)


# STREAMLIT APP

st.title("Albot's Movie Recommender")
st.markdown(
    """
    Do you feel lucky, punk? Frankly, my dear, I don't give a damn. I'm the captain now. 
    I'm going to make you an offer you can't refuse. May the odds be ever in your favour.
    """
)

## INPUT

input_titles = st.multiselect(
    "Choose your favorite movies:",
    movies["title"]
)
input_ids = [id for id in movies[movies["title"].isin(input_titles)]["movie_id"]]


## RECOMMENDATIONS

if "recommendations_given" not in st.session_state:
    st.session_state.recommendations_given = False

if st.button("Make my day"):
    
    try:
        cf_recommendations = cf_recommender.get_recommendations(input_ids, 10)
        cbf_recommendations = cbf_recommender.get_recommendations(input_ids, 10)

        st.write("**Collaborative filtering recommendations:**")
        st.dataframe(cf_recommendations)

        st.write("**Content-based filtering recommendations:**")
        st.dataframe(cbf_recommendations)

        st.session_state.recommendations_given = True

    except ValueError:
        st.warning("Please select at least one movie.")


## ASKING FOR VALIDATION

like_responses = [
    "I think this is the beginning of a beautiful friendship.",
    "I like you very much. Just as you are.",
    "You had me at hello.",
    "Here's looking at you, kid.",
    "Merry Christmas, ya filthy animal.",
]
hate_responses = [
    "Why so serious?",
    "It's just a flesh wound.",
    "Yeah, well, that's just, like, your opinion, man.",
    "My mama always said, life was like a box of chocolates. You never know what you’re gonna get.",
    "You can’t handle the truth!",
    "Bye, Felicia.",
    "You talkin' to me?"
]

if st.session_state.recommendations_given:

    st.write("**Did you like my recommendations?**")
    left, right = st.columns(2)

    if left.button("For sure.", use_container_width=True):
        seed = np.random.randint(0, len(like_responses))
        left.write(f"{like_responses[seed]}")

    if right.button("Hell no!", use_container_width=True):
        seed = np.random.randint(0, len(hate_responses))
        right.write(f"{hate_responses[seed]}")