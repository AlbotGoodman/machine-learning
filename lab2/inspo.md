# Requirements

Choose your feature space and method for redcommendation. Then implement a system that recommends five movies given a certain input. 

1. Predictive rating
- regression on ratings
- most suited for content filtering:
    - training on similar users the preditions might be good enough
    - one row per user
    - filter data (e.g."rating experts" > 100 ratings/user) to minimise dimensionality

2. Similarities in genres
- one-hot encoding on genres
- KNN-Transform with cosine similarity as metric

3. NLP
- combine tags and genres
- use something like TF-IDF to capture similarities between movies

4. Combination approach
- e.g. 1+3

5. KMeans
- clustering around the recommended movies to increase diversity (?)

# Ideas

- sentiment analysis on tag
- find positive/negative tags by rating
- sentiment profile for each movie
    - as to recommend movies with similar sentiment
- combine sentiment features with genre and rating data
- use PCA to reduce dimensionality
- use different models for different recommendations and use them all combined with a voting classifier
- there should be a fallback system (not all movies have tags nor ratings but they all have genres)
    - perhaps sentiment analysis together with ratings will give the best results
    - but if an input movie has no tags then ...
        - movies contain the most movies
        - it also contains genres
        - from genres we can get similar movies 
        - perhaps the user can vote on which of the five suggestions are the most similar
        - then we can see if that one together with the first input might get better results

## Tags

- reduce the number of rows by combining tags from the same user regarding the same movie
- regularisation, is that any good here? 

## Ludwilton

- begränsa antalet recensioner per film till 100 (utan att filtrera bort filmerna)
- spara modellerna med joblib för att inte träna igen
- ludde såg förbättring med >200 c_components för NMF trots att elbow plot inte visade på förbättring
- kanske ta bort de användare med standardavvikelse nära noll, inte bara noll ALTERNATIVT ta bort alla noll i stället för att sätta till 0.5?
- testa större modell men mindre data för att se (som jag gjorde i förra labben)