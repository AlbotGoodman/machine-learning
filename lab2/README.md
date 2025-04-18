# Laboration 2 - Movie Recommendation System

## Introduction

This laboration is part of a machine learning course at [ITHS](www.iths.se), taught by [Raphael Korsoski](www.github.com/pr0fez). The purpose of the lab is to manipulate data from the the [movielens dataset](www.grouplens.org/datasets/movielens/) and through various méthods create a system that, given an input, recommends a handful of movies. A combination of content-based and collaborative filtering is used in conjunction to get the best results. 

In the dataset there are three main files used:

- movies.csv
- ratings.csv
- tags.csv

The user and movie IDs are consistent through the different files.

## Content-based filtering

From movies.csv there is a column with genres per movie and in tags.csv there is a column containing user generated keywords (tags) per movie. These are combined into one semantic corpus. The content of this new column is preprocessed and vectorised before a topic modelling technique called LDA (Latent Dirichlet Allocation) is used. It identifies $N$ number of latent topics (or themes) from the corpus. Each movie will then contain a distribution of each topic. Similarity is calculated on topic distributions. 

## Collaborative filtering

From ratings.csv the user generated ratings per movie is used. Since the ratings are not normally distributed the data was standardised per user. These new ratings are then used to create a sparse user-movie matrix. Just like the previous filtering method latent factors are extracted here as well through matrix factorisation where mini-batch gradient descent is used for optimisation. 