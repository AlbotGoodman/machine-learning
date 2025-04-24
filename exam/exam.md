# 1

$ Precision = \frac{TP}{TP + FP} $  
$ Recall = \frac{TP}{TP + FN} $  

## a)

Klass A:

$ Precision = \frac{67}{67 + 9} = 0.88 $
$ Recall = \frac{67}{67 + 68} = 0.50 $

## b)

Klass B:

$ Precision = \frac{60}{60 + 68} = 0.47 $
$ Recall = \frac{60}{60 + 9} = 0.87 $

# 2

$ ? Hyperparameter = \đrac{1}{n} $  
$ SSE = \sum^n_{i=1} (y_i - ŷ)^2 $  

Tillsammans utgör dessa två MSE. 

$ Ridge Regression Hyperparameter = \lambda $  
$ Ridge Regression = \sum^p_{j=1} \beta^2_j $  

# 3

Vid oövervakad inlärning saknas generellt en responsvariabel/etikett och metoden som används försöker i stället utröna mönster i datan, t ex vid klustring (K-Means) eller dimensionsreducering (PCA). Ett tillfälle då träningsdata implicit används är vid hyperparameteroptimering med korsvalidering. 

# 4

Vid logistisk regression finns det hyperparametrar som optimeras vid korsvalidering, till exempel regulariseringsparametern lambda (även kallad alpha i sklearn) som reglerar styrkan av Lasso, Ridge eller ElasticNet. 

# 5

Medam Ridge hade fördelat variansen över variablerna kan Lasso sätta små värden till noll för variabler som inte tillför något till regressionen (eller klassificeringen vid t ex logistisk regression). 

# 6

Figuren visar ett icke-linjärt förhållande i datan vilket är omöjligt för en regressionslinjen att förutsäga. Vi har alltså en kraftig underfit där modellens förklaringsgrad är väldigt låg. 

# 7

De primära hyperparametrarna i kostnadsfunktionen är lambda och alpha. Vad vi ser är logistisk regression som använder MSE och ElasticNet som regulariseringsmetod. 

# 8

Jag hade valt en SVM för klassificering med soft margin classifier på grund av överlappningen som finns i datan. Det ser ändå ut att finnas en någorlunda klar skiljelinje mellan klasserna och med en soft margin kan vi tillåta en viss felmarginal, att flera datapunkter hamnar innanför våra support vectors. Eftersom nämnd skiljelinje ser någorlunda rak ut kan vi nog få ganska bra resultat oavsett vilken kernel vi använder. För att fånga så komplexa mönster som möjligt med hjälp av oändligt antal dimensioner hade jag valt radial basis function som kernel men hade lagt noga vikt vid hyperparameteroptimering för att försöka undvika overfit. 