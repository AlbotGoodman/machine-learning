# Viktiga ämnen inför tentan

- **Klassificeringsmatriser**: skriv upp definitionerna för de olika mätvärdena (accuracy, recall, precision). Öva på att räkna med två eller tre klasser. 
- **Kostnadsfunktioner**: MSE, regularisering, logit
- **Hyperparametrar**: för regression och klassificering; särskilt för regularisering och logistisk regression
- **Korsvalidering**
- **Klustring och PCA**: vad används de till och hur?
- **Splits**: tränings-, validerings- och testdata; när används vilken sorts data? 
- **SVM och kerneltrick**: varför; vad åstadkommer det?
- **Beslutsträd**: dess hyperparametrar och begränsningar för de olika sorterna
- **Klassificering vs värdesregression**: vad är skillnaderna; hur används de; vilka metoder kan blanda värden och kategorier? 

# Svar

## Classification Matrices

Classification matrices (confusion matrices) are fundamental for evaluating the performance of classification models.

Key metrics:

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

- Proportion of correct predictions among all predictions
- Good for balanced datasets, misleading for imbalanced ones

**Precision** = TP / (TP + FP)

- Proportion of true positive predictions among all positive predictions
- "When the model predicts positive, how often is it correct?"
- High precision means low false positives (important when false positives are costly)

**Recall** (Sensitivity) = TP / (TP + FN)

- Proportion of true positives among all actual positives
- "How many actual positives did the model correctly identify?"
- High recall means low false negatives (important when missing positives is costly)

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

- Harmonic mean of precision and recall
- Useful when you need to balance precision and recall

For multi-class problems, these metrics are calculated per class (often using "one-vs-rest" approach) and then averaged.

Remember that choosing the appropriate metric depends on your problem context - for example, in medical tests, you might prioritize recall over precision.

## Cost Functions

Cost functions measure how well your model is performing by quantifying the difference between predicted and actual values:

**Mean Squared Error (MSE)** = (1/n)∑(y - ŷ)²

- Used for regression problems
- Heavily penalizes large errors due to squaring

**Mean Absolute Error (MAE)** = (1/n)∑|y - ŷ|

- Also for regression
- More robust to outliers than MSE

*Regularization terms:*

**Ridge (L2)**: λ∑(θᵢ)²

- Shrinks all coefficients toward zero
- Good for handling multicollinearity

**Lasso (L1)**: λ∑|θᵢ|

- Can shrink coefficients exactly to zero (feature selection)

**Elastic Net**: Combination of L1 and L2

**Logistic Loss/Log Loss** = -[y·log(p) + (1-y)·log(1-p)]

- Used for binary classification

Understanding these functions is crucial as they define what your model is optimizing for, and different cost functions lead to different model behaviors.

## Hyperparameters

Hyperparameters are configuration settings that aren't learned from data but set prior to training:

**For regression models:**

- Learning rate in gradient descent
- Regularization strength (λ in Ridge/Lasso)
- Polynomial degree in polynomial regression

**For classification:**

- k in k-nearest neighbors (L6-KNN.ipynb)
- C (regularization parameter) in SVMs (L8-SVM.ipynb)
- max_depth, min_samples_split in decision trees (L9-Decision_tree.ipynb)
- n_estimators in random forests (Lec10-RandomForest.ipynb)

**Tuning approaches:**

- Grid search: trying all combinations of predefined parameter values
- Random search: sampling from parameter distributions
- Cross-validation is essential for proper tuning

Good hyperparameter selection can dramatically improve model performance, while poor choices can lead to underfitting or overfitting.

## Cross-validation

Cross-validation is a robust technique for evaluating model performance by using multiple train/test splits:

**k-fold cross-validation:** 
Data is split into k equal parts (folds); model is trained k times with different folds as validation sets

**Leave-one-out cross-validation:** 
Extreme case where k equals number of samples

**Stratified k-fold:** 
Maintains class distribution in each fold

*Key benefits:*

- Reduces dependence on a single train/test split
- Provides more reliable performance estimates
- Essential for hyperparameter tuning
- Helps prevent overfitting

Scikit-learn's GridSearchCV combines cross-validation with hyperparameter tuning.
Remember that while cross-validation is powerful, it's computationally expensive since you're training multiple models.

## Clustering and PCA

Clustering (unsupervised learning) groups similar data points:

**K-means clustering:**

- Partitions data into k clusters
- Each point belongs to cluster with nearest mean
- *Process:* 
    (1) Initialize k centroids, 
    (2) Assign points to nearest centroid, 
    (3) Update centroids, 
    (4) Repeat until convergence
- Challenge: choosing optimal k (elbow method, silhouette score)
- Applications: Customer segmentation, image compression

**Principal Component Analysis (PCA):**

- Dimensionality reduction technique
- Transforms data to new coordinate system where axes (principal components) capture maximum variance
- Helps with visualization, noise reduction, feature extraction
- *Key concepts:*
    Principal components are eigenvectors of the covariance matrix
    Explained variance ratio helps determine number of components to keep

Both techniques help us understand structure in data when we don't have labels. PCA is particularly useful before applying supervised learning to high-dimensional data.

## Splits

Proper data splitting is crucial for model validation:

**Training set:** Used to fit the model (typically 60-80% of data)
**Validation set:** Used to tune hyperparameters
**Test set:** Used for final evaluation; should be touched only once

**Key points:**

- Split data before any preprocessing to prevent data leakage
- For time series data, use chronological splits
- Stratified sampling maintains class distribution across splits

**Common pitfalls:**

- Data leakage: when test information influences training
- Using test data repeatedly leads to overfitting to the test set
- Not having enough data in each split

The distinction between validation and test sets is critical - the validation set helps you tune your model, while the test set gives you an unbiased estimate of your final model's performance.

## SVM and Kernel Trick

Support Vector Machines find the optimal hyperplane that maximizes the margin between classes:

**Maximum margin classifier:** Works for linearly separable data
**Soft margin classifier:** Allows some misclassifications for better generalization
**Support vectors:** Points closest to the decision boundary

**Kernel trick:**

- Implicitly maps data to higher dimensions without computing the transformation
- Enables separation of non-linearly separable data
- Common kernels:
    Linear: K(x,y) = x·y
    Polynomial: K(x,y) = (γx·y + r)^d
    RBF (Gaussian): K(x,y) = exp(-γ||x-y||²)

**Key hyperparameters:**

- C: Controls regularization (smaller C = more regularization)
- Kernel type and parameters (γ, d)

SVMs are powerful for both classification and regression, particularly effective for complex but small to medium-sized datasets.

## Decision Trees

Decision trees recursively split data to create homogeneous subsets:

**Key characteristics:**

- Intuitive interpretation
- Can handle both numerical and categorical data
- Prone to overfitting without pruning

**Hyperparameters:**

- max_depth: Limits tree depth
- min_samples_split: Minimum samples required to split
- min_samples_leaf: Minimum samples in leaf nodes
- criterion: Splitting criterion (Gini, entropy for classification; MSE for regression)

**Limitations:**

- High variance (sensitive to data variations)
- Can create biased trees if classes are imbalanced
- Often outperformed by ensemble methods

**Ensemble improvements:**

- Random Forest: Multiple trees on bootstrapped samples with random feature subsets
- Boosting methods: AdaBoost, Gradient Boosting

Decision trees provide the foundation for many powerful ensemble methods that are among the most successful ML algorithms.

## Classification vs. Value Regression

*Key differences:*

**Classification:**

- Predicts discrete categories/classes
- Evaluation: accuracy, precision, recall, F1-score, ROC-AUC
- Output: class labels or probabilities
- Examples: logistic regression, decision trees, SVM, naive Bayes

**Value Regression:**

- Predicts continuous values
- Evaluation: MSE, MAE, R², RMSE
- Output: numerical predictions
- Examples: linear regression, regression trees, SVR

**Mo6dels that can handle both:**

- Decision trees/Random forests (with appropriate criterion)
- SVMs (SVR for regression)
- Neural networks (with appropriate output layer and loss function)

**Feature handling:**

- Both may require feature scaling
- Classification often needs encoding for categorical features
- Regression might benefit more from outlier removal