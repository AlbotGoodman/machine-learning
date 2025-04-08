# K-Nearest Neighbour

*KNN is a versatile non-parametric method used for both classification and regression.*

## Classification

Given a new data point, KNN looks at the k closest data points to determine the type of the new data point. The k can be any number, preferrably an uneven one. It determines the type (or class) by majority vote. Say k=11 and the new data point has 7 red, 3 blue and 1 green neighbours the majority is red and therefore the new entry is classified as red as well. 

Low values for k can be noisy and subject to the effects of outliers (*overfitting*). Large values for k can "vote off" smaller classes, e.g. k=33 and the new point is in a red cluster but red only has 11 entries then we incorrectly label the new data point as something else (*underfitting*). 

## Regression

For regression KNN instead averages the values of the k nearest neighbours. For example, if k=3 and the three closest data points are [3.5, 6.5, 5.0] then the new data point would be the mean of these three, i.e. 5.0. 

## Misc

Even though we can use train/test split with KNN it doesn't learn anything. Instead it just picks what is most similar to previous examples. 

- can use different distance metrics (euclidean, manhattan etc)
- makes no assumptions about the distribution
- scaling is necessary