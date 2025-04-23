# Natural Language Processing

*NLP involves processing and analysing human language data*

Absolutely necessary for NLP is a varying amount of text preprocessing. 
- lowercase conversion
- stemming/lemmatisation
- stopwords removal
- other (remove links, words with numbers etc)

## TF-IDF

$$tf(t,d) = \frac{f_{t,d}}{\sum_{t'\in d} f_{t',d}}$$  

**Term Frequency** is how often a term (word) occurs in a document (collection of words). Basically the formula translates to the total number of a specific word divided by the total number of words. 

$$idf(t,D) = \log{\frac{|D|}{1+|\{d\in D: t\in d\}|}}$$  

**Inverse Document Frequency** gives information on the rarity of the word in all documents. Basically the formula translates to the logarithm of the total number of documents divided by the number of documents with the term present. 

$$tfidf(t,d,D) = tf(t,d)\cdot idf(t,D) $$  

We combine these two by multiplying them to get TF-IDF. Now let's explain this using an example where we find out what ingredients are most unique and important when baking pancakes. 

$d_1$ (*regular pancakes*) contains the following ingredients: 
- milk
- flour
- eggs
- butter
- sugar
- salt
- vanilla extract

$d_2$ (*american pancakes*) contains the following ingredients: 
- milk
- flour
- eggs
- butter
- sugar
- baking soda
- syrup

$d_3$ (*banana pancakes*) contains the following ingredients: 
- milk
- bananas
- eggs
- oatmeal
- cinnamon
- honey

The TF of milk in $d_1$ is $1/7 \approx 0.14$.  
The IDF of milk is $log(3/3) = 0$.  
The TF-IDF of milk is $0.14 \cdot 0 = 0$.  

The TF of syrup in $d_2$ is $1/7 \approx 0.14$.  
The IDF of syrup is $log(1/3) \approx 0.48$.  
The TF-IDF of syrup is $0.14 \cdot 0.48 = 0.0672$.