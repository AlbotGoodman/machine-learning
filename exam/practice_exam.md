# Practice exam

## Q1

|   | A | B |
|---|---|---|
| A | TN| FP|
| B | FN| TP|

### a)

$$ precision = \frac{TP}{TP+FP} = \frac{1030}{1030 + 37} \approx 0.965 $$  

$klassificering_1$ har en precision score på 96,5 %. 

$$ precision = \frac{TP}{TP+FP} = \frac{980}{980 + 0} = 1 $$  

$klassificering_2$ har en precision score på 100,0 %. 

*Svar: Klassificering 2 har högst precision med avseende på klass B.*

### b)

$$ recall = \frac{TP}{TP+FN} = \frac{1030}{1030 + 0} = 1 $$  

$klassificering_1$ har en recall score på 100,0 %. 

$$ recall = \frac{TP}{TP+FN} = \frac{980}{980 + 43} \approx 0.958 $$  

$klassificering_2$ har en recall score på 95,8 %. 

*Svar: Klassificering 1 har högst recall med avseende på klass B.*

### c)

B = aktivera självförstörelsemekanism

*Svar: Om klass A innebär att allt är okej medan klass B innebär att självförstörelsemekanismen måste aktiveras så är klassificering 1 det självklara valet. Vid 100 % av alla tillfällen med klass B så klassificerade systemet korrekt klass B. Recall är alltså det bättre valet att optimera för när vi inte får missa positiva fall.*

## Q2

### a)

$X_1 = 1$  
$X_2 = 2$  

$\beta_0 = 1$
$\beta_1 = 1$
$\beta_2 = 3$
$\beta_3 = 2$

$Y = 1 + 1 \cdot 1 + 3 \cdot 2 + 2 \cdot 1 \cdot 2$  
$Y = 12$

*Svar: Värdet för punkten (1, 2) är 12.*

### b)

*Svar: Linjen f är linjär.*

## Q3

### a) 

*Svar: Datamängden är linjärt separerbar i en högre dimension men inte med en linje i 2D. Eftersom vi har en tvådimensionell graf med cirklar och kryss kan vi tyda att vi har tre dimensioner eftersom datapunkterna på grafen skiljer sig (kryss/cirkel). Därför skulle vi i en högre dimension kunna skilja datamängden med ett hyperplan.*

### b) 

*Svar: Datapunkterna som är representerade med kryss sträcker sig längs ena diagonalen av grafen medan datapunkterna som är representerade med kryss sträcker sig längs den andra diagonalen. Kryssen har en större variation, större spridning längs sin diagonal än vad cirklarna har på sin diagonal. Därför hade en regressionslinje längs kryssen varit vår första principalkomponent medan vår andra hade varit längs cirklarna. På så sätt kan vi förklara majoriteten av spridningen i två dimensioner i stället för tre. Vi har reducerat antalet dimensioner.*

## Q4

*Svar: $\bold{Y} = \bold{X}\beta$ är en rät linje. Om värdena på lutningen (slope) och skärningen (intercept) är anpassade utifrån träningsdata finns det en risk att den är overfit; antingen för att vi hade för få träningspunkter eller att det finns brus i datan som algoritmen lärt sig. För att få modellen att generalisera bättre introducerar vi en bias-term. Regulariseringsparametrar påverkar hur mycket bias som introduceras. Lasso tenderar att sätta parametrar till noll om de inte är viktiga medan Ridge sprider bruset över alla parametrar.*

## Q5 

*Svar: 1) Då regressionen lägger för stor vikt vid vissa parametrar kan utvecklaren tjäna på att använda antingen Lasso eller ElasticNet i stället för Ridge. 2) Eftersom regressionen får oväntat dåligt resultat i jämförelse med förväntningar från tidigare samlad statistik kan det hända att modellen påverkats av outliers. Då kan MSA (absolutbeloppet) användas i stället för MSE.*

## Q6

*Svar: En train/test split är inte relevant vid 1) oövervakad inlärning då vi inte har kända etiketter (responsvariabler) i datan, och 2) vid en cross-validation då den automatiskt delar upp datan i flera train/test splits och validerar på alla i tur och ordning.*

## Q7

*Svar: Random Forest är en bättre metod att använda då den kan utnyttja parallellism, mindre risk för overfit och kan delas upp i flera stubbar vilket minskar komplexiteten.*

## Q8

*Svar: Generellt sett är en SVM snabbare då den inte beräknar koordinater för de högre dimensionerna utan i stället använder kerneltricket och beräknar en relationsfaktor för avståndet mellan parametrar i stället. Däremot är den inte lika enkel att tyda för statistiska syften, då kanske expansionen är att föredra. Om resurser finns kan först en expansion användas följt av en kernel.*