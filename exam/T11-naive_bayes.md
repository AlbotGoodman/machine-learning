# Naive Bayes

*Naive Bayes classifiers apply Bayes' theorem with "naive" independence assumptions between features.*

$$ P(A|B) = \frac{P(B|A) P(A)}{P(B)} $$  

Like LDA and QDA, Naive Bayes uses **Bayes Theorem** but in a distinctly different approach. Here we no longer make any assumptions on the distribution but another assumption: "within each class, p predictors are independent." By assuming independence, we can focus solely on estimating the marginal distribution of each predictor separately, which is much simpler.

In essence, Naive Bayes calculates the probability of an event occurring given the presence of certain features. It uses this probability to classify new data points into different categories.

This can be very useful with high-dimensional data as in text classification. If the naive assumption is proven incorrect the Naive Bayes will perform badly - which is in itself a hint on correlation between variables. 