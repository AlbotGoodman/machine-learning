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