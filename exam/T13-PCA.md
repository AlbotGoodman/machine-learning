# Principal Component Analysis

*PCA reduces dimensionality by transforming data to a new coordinate system of orthogonal components.*

The main purpose behind PCA is to find uncorrelated variables that explain as much of the variance as possible (via linear combinations). To do this it's essential to standardise the data. If we can find components that explain the majority of the variance then we don't need all dimensions. In that sense PCA is an unsupervised learning technique for reducing dimensionality. 

By reducing the number of dimensions we decrease the amount of data (but not too much or valuable data) and speed up computations. With fewer dimensions there is also less noise (usually spread out among dimensions) and we prevent overfitting. When doing regression, classification or clustering the PC scores can be used instead of the original features since it leads to less noisy results. 

The principal component vectors are eigenvectors of the covariance matrix of the feature matrix. The first PC has the largest possible variance among all linear combinations. We can also say that it has the largest sum of squared distances to the best fitting line through the data (the line made up of linear combinations). While the average of the sum of squared distances is called eigenvalue. 

The second PC has the largest possible variance that is uncorrelated with PC1. In other words it's orthogonal (perpendicular, 90 degrees). The number of PCs we have is either the number of variables or samples, whichever is smallest. But the necessary number of PCs to explain the majority of the data is determined via a scree plot where we look for an elbow where there are diminishing returns for more PCs. Imagine this: 

- PC1 accounts for 78 % of the variation in the data
- PC2 accounts for 16 % of the variation in the data
- PC3 accounts for 4 % of the variation in the data
- PC4 accounts for 2 % of the variation in the data

Then PC1 and PC2 would account for 94 % of the data and we wouldn't need the third or fourth PC. Now we can visualise the data in a 2D graph where as with 4D we could not! 