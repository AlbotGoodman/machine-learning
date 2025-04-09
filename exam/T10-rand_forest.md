# Forests (or ensembles)

*Random forests combine multiple decision trees to improve prediction accuracy and control overfitting.*

For ensemble methods there are mainly three methods that improve upon decision trees:

- bagging, or **b**oost **agg**regation
- random forests 
- boosting

These methods doesn't need to be scaled but we need to HPO. Be careful not to use too many (combinatory explosion). Also they should not be used in critical systems, according to Raphael. 

## Bagging

This reduces variance in *regression* by averaging the predictions from multiple trees and for *classification* use majority vote. The trees are grown deep without pruning (high variance, low bias) and then averaging predictions (from random batches) to reduce variance. Quite similar to cross validation. The importance of each variable can be determined by measuring the total decrease in RSS (regression) or gini index (classification). 

## Random Forests

Only a subset of predictors are used for each tree. This decreases correlation randomly and makes strong predictors less dominant for prediction/classification. 

## Boosting

It combines multiple weak learners into a strong learner. The process works by training models sequentially, where each new model focuses on the errors made by the previous models, thereby improving the accuracy of the system as a whole. This often works better with smaller trees. 