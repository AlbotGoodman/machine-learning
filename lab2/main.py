# DEPENDENCIES

import streamlit as st
# from collaborative_filtering import *
# from content_based_filtering import *
import pandas as pd
import numpy as np
st.set_page_config(page_title="AsMR")


# LOAD DATA

@st.cache_data
def load_data():
    """
    Load data from CSV files and return DataFrames.
    """
    ratings = pd.read_csv("../data/movielens/ratings.csv")
    movies = pd.read_csv("../data/movielens/movies.csv")
    tags = pd.read_csv("../data/movielens/tags.csv")
    return ratings, movies, tags


ratings, movies, tags = load_data()


# DATA PREPROCESSING

pass


# STREAMLIT APP

st.title("Albot's Movie Recommender")
st.markdown(
    """
    Do you feel lucky, punk? Frankly, my dear, I don't give a damn. I'm the captain now. 
    I'm going to make you an offer you can't refuse. May the odds be ever in your favour.
    """
)

input_list = st.multiselect(
    "Choose your favorite movies:",
    movies["title"]
)

if input_list == False:
#     st.write("")
# else:
    st.write("Please select at least one movie.")


## RECOMMENDATIONS

pass


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

st.write("**Did you like my recommendations?**")
left, right = st.columns(2)

if left.button("For sure.", use_container_width=True):
    seed = np.random.randint(0, len(like_responses))
    left.write(f"{like_responses[seed]}")

if right.button("Hell no!", use_container_width=True):
    seed = np.random.randint(0, len(hate_responses))
    right.write(f"{hate_responses[seed]}")