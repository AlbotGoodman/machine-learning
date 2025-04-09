# Regularisation

*Regularization techniques help prevent overfitting by adding penalty terms to the model's complexity.*

## Linear models
Remember that in linear models we use Ordinary Least Squares (OLS) which finds the minimum sum of squared errors (SSE from the statistics course).

$$ \text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

A regression line using OLS reflects the relationship between the variable on the x axis and the variable on the y axis.

$$ y = \beta_0 + \beta_1X_1 $$

However when there are few test points in the data - for example in the training split - the regression line doesn't necessarily reflect new data points (unseen in training) very well. In this moment the line is overfit. 

$$ \text{underfit} = \text{low variance} + \text{high bias} $$  
$$ \text{overfit} = \text{high variance} + \text{low bias} $$  

$$ \text{high variance} = \text{low precision} $$  
$$ \text{high bias} = \text{low accuracy} $$  

Remember: if you have good enough results them you don't need to regularise.

## Ridge Regression (l2-norm)
When a model is too complex, the dataset too small and noisy the model will pick up those patterns and result in an overfit. 

The main idea with Ridge Regression is that we doesn't want an overfit so we introduce bias. At first that makes the line not fit the training data very well but better captures the relationship in unseen data - it generalises better. 

We can use bias to lower the variance to provide a better long term predictions. 

$$ \text{SSE} + \lambda (\beta_1^2 + \beta_2^2 + ... + \beta_n^2) $$

While OLS minimises SSE, Ridge Regression minimises SSE plus a penalty term (lambda) multiplied by each parameter (X). The larger we make lambda the closer to a horisontal line we get (but never truly horisontal) i.e. decreasing variance. Ridge Regression spreads the variance between all parameters.  Lambda is determined during cross validation (then called alpha). 

$$ \text{bias} > \text{variance} $$

Also, when there is not enough data to find OLS parameter estimates, Ridge Regression can still find a solution with cross validation and the Ridge Regression penalty. 

Here is the cost function of Ridge Regression: 

$$ C(\vec{\theta}) = MSE(\vec{\theta}) + \lambda \frac{1}{2}\sum_{i=1}^n \theta_i^2 $$

When we regularise we change the cost function and penalise the solution that we don't want. 

## Lasso Regression (l1-norm)
This is very similar to Ridge Regression, it also tries to minimise variance by introducing bias. Before we squared the parameters but here we take the absolute value instead. 

$$ \text{SSE} + \lambda (|\beta_1| + |\beta_2| + ... + |\beta_n|) $$

When we before increased the penalty term (lambda) we got an almost horisontal line. Lasso Regression instead eliminates variables entirely. That means that instead of spreading variance between variables/coefficients it can set their values to zero. That way it excludes variables which makes it a little better att reducing variance in models that may contain a lot of "useless" variables. In contrast, Ridge Regression tends to do better when most variables are useful. 

Here is the cost function of Lasso Regression: 

$$ C(\vec{\theta}) = MSE(\vec{\theta}) + \lambda \frac{1}{2}\sum_{i=1}^n |\theta_i| $$

## Elastic Net (both)

$$C(\vec{\theta}) = MSE(\vec{\theta}) + \lambda\left(\alpha\sum_{i=1}^n |\theta_i| + \frac{1-\alpha}{2}\sum_{i=1}^n \theta_i^2\right)$$  

Lasso is valuable for feature selection, while Ridge is often better when all predictors contribute or when dealing with multicollinearity. 