import streamlit as st
# from collaborative_filtering import *
# from content_based_filtering import *

st.set_page_config(page_title="AsMR")
st.title("Albot's Movie Recommender")
st.markdown(
    """
    Dear visiting cinephile,

    I heartily herald thee to my revered repository of radiant reels. Rest assured, thy requests for remarkable recommendations shall be readily received and rigorously rendered.

    Yours sincerely,  
    Albot Goodman
    """
)
input_list = st.multiselect(
    "Choose your favorite movies:",
    ["Harry Potter 1", "Harry Potter 2", "Harry Potter 3", "Harry Potter 4", "Harry Potter 5", "Harry Potter 6", "Harry Potter 7", "Harry Potter 8"]
)

if input_list:
    st.write(f"You selected: {', '.join(input_list)}")
else:
    st.write("Please select at least one movie.")