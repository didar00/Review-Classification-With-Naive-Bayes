# Report

## Naive Bayes Classifier

With the help of the assumption that the words in a sentence can be taken into consideration independently, Naive Bayes classifier is convenient for text classification. All we need to do is calculate mutually independent probabilities of each word in the sentence and compare them without the need of normalization.

![naive_formula.png](attachment:naive_formula.png)

To achieve this goal, probabilities of two classes (positive and negative in this assignment) must be calculated for each sentences.

## Preparation of Input

Before getting into the classification algorithm, input must be cleaned and prepared to avoid the unnecessery calculations and to satisfy the required result.

*camera pos 69.txt i recently purchased this for use with my 550ex . if you use alot of flash for event photography , you will want this in your kit . it is expensive , but i think it is worth every penny . it will keep you from changing aas in your strobe and it quickens the recycle times to nearly instant . you can shoot bursts and still get the proper flash exposure . you can attach a strap to sling it over your shoulder or slip it on your belt . mounting on my 20d was just not comfortable . it gets in the way of the hand-grip *

For each review **(one raw review input can be seen above)**, punctuations and numerical values were removed throughout the program. Each input review was stored under a dataframe which consists of 4 columns: category, review class, identifier (which was stored as plain numbers: 345 i.e.) and the review itself.
Stop words, both was included and excluded to understand the effect of it.
**(processed review input below)**

*camera pos 69 i recently purchased this for use with my ex  if you use alot of flash for event photography you will want this in your kit it is expensive but i think it is worth every penny it will keep you from changing aas in your strobe and it quickens the recycle times to nearly instant you can shoot bursts and still get the proper flash exposure you can attach a strap to sling it over your shoulder or slip it on your belt mounting on my d was just not comfortable it gets in the way of the hand grip*

## Bag of Words

In order to calculate the probability of a given word in a sentence, given a class, probability of the word must be known. This can be calculated from the frequency of the word in that given class. To ease that process, since it may be used for hundreds of words, **bag of words** method was used. In **BoW**, every word in the corpus is stored along with the frequency information. However, in case of absence of a word in the BoW ( zero frequency), the total probability becomes zero whatever the other probabilities are. Therefore, **laplace smoothing** was used to avoid zero probability.

![laplace.png](attachment:laplace.png)

### Unigram

First way to implement Naive Bayes classifier to texts is to approach words singularly. It is a helpful and easy way to classify the texts. However, in the process of singularity, the meaning of word groups (the word that their meaning depends on the surrounding words) is excluded.

### Effect of Stop Words in Unigram

At the first step, accuracy was calculated with stop words included. The accuracy with stop words is **0.809**.
The next step, stop words was not taken into consideration. The accuracy is **0.801**.
In this case, the change in the accuracy in the level of **%0.8**, which is a non significant ratio to compare the cases.

### Bigram

In contrast to unigram BoW, bigram enables to keep the meanings of the word pairs. In this case, BoW composed of word pairs and their frequencies in the corpus. Every other process are the same as unigram implementation. However, meanings that come from word pairs are not excluded and helps to classify the text more accurately.

### Effect of Stop Words in Bigram

Without stop words in bigram the accuracy is **0.734**. Without leaving out the stopwords, bigram has showed a slightly better performance with accuracy of **0.749**.

The effect of leaving out stop words on accuracy may change from dataset to dataset. In this dataset, leaving stop words out has improved the accuracy slightly since they are so common but give so little information about the classification. However, there may be cases that removing stop words improves performance if the stop words dictionary is expanded with the well-processed not-important words. 

## Accuracy Analysis

Algorithm has been trained with the **%80** of the data and has been tested with the **%20** of the data.
The accuracy of the model is **%81.9**.
To visualize the accuracy of the algorithm, **confusion matrix** can be seen below.

![confusion.png](attachment:confusion.png)

## Error Analysis

Number of false positives is 236, and number of false negatives is 254. That gives us the error of **%18** in the process of classifying. These misclassifications may be caused from the data that has not been cleaned enough from the meaningless words even though the vocabulary was restricted throughout the process to improve the accuracy. In some cases, bigram remains incapable to store the meanings of words group which are consist of more than 2 words. N-gram bag of words can be used interchangibly to improve the performance further. 

## Module Analysis

Not every word has the same importance when it comes to giving a direction to the classifier to classify the text. Some words reveal much more information about the class of the text. For example, word "excellent" is more helpful than "bookshop" in terms of understanding the emotion of the text. Therefore has more importance.
In order to find these words, **term frequency-inverse document frequency** can be used. In the light of this, each term has a frequency which is expressed as the number of times each term occurs in each document. On the other side, inverse document frequency gives the information of how common or rare a word is in the entire corpus. Multiplication of these terms results in the **TF-IDF** score of a word in a document. 

![tf_formula.png](attachment:tf_formula.png)

![idf_formula.png](attachment:idf_formula.png)

![tf_idf.png](attachment:tf_idf.png)




**The most important words that signals the review is positive are:**  
*great    0.238029  
like     0.219102  
just     0.196940  
good     0.192747  
time     0.154773  
really   0.134528  
best     0.118476  
love     0.116798  
way      0.087449  
better   0.085293*  

**10 words whose absence predicts the review is positive:**  
*bad       0.082980  
people    0.081414  
know      0.076568  
money     0.075973  
used      0.067214  
think     0.066642  
problem   0.064126  
little    0.059674  
support   0.058570  
actually  0.053789* 

**The most important words that signals the review is negative are:**  
*bad       0.082980  
people    0.081414  
new       0.079296  
money     0.075973  
bought    0.074671  
got       0.071067  
problem   0.064126  
say       0.063191  
little    0.059674  
support   0.058570*  

**10 words whose absence predicts the review is negative:**  
*great   0.170570  
like    0.146257  
good    0.136429  
time    0.120247  
best    0.110028  
love    0.105577  
does    0.098961  
new     0.090242  
easy    0.083994  
years   0.078169*

## Restricting the Vocabulary Space

To improve the accuracy, text classification can be completed through the restricted space of vocabulary which consists of only most informational words. This can be achieved through using **TF-IDF** scores of the words in the required sentence.  
After this improvement, the accuracy with **unigram** increased by **%1** from **%81.9** to **%82.7**; with bigram decreased by **%2.1** from **%73.4** to **%71.3**.  
These decrease in the bigram can be explained as the loss of meaning that comes from words pairs since we took out some of the words. On the other side, removing unnecessary words helped to increase the accuracy with unigram implementation.
