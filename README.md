# AmazonReviews_NLP 

## Introduction: 
Opinions shared in the form of reviews on platforms like Amazon have been impacting the sale of certain products. In the ecommerce
and digital engagement era, people got a free platform to express their opinions in the form of reviews which have become a crucial aspect of making business decisions. Customers usually refer to the review before buying any product, especially electronics. Therefore, analyzing the reviews to gain insights will help businesses and ecommerce websites like
Amazon improve their recommendation system. In the project, customer review analysis considered electronic product reviews in Amazon to get insights from reviews and build a word cloud using two different Vectorization methods of NLP and applying ML model classification, which will help understand various product categories and how they are doing in market.

The review ratings have been segmented into positive and negative categories. The text within the reviews has undergone preprocessing utilizing the 'nltk' toolkit and Spacy, involving processes such as tokenizing words, eliminating stop words, and performing both stemming and lemmatization. This prepared data was then subjected to a variety of machine learning techniques, focusing on feature extraction methods such as part-of-speech tagging, word frequency analysis, TFIDF, and the Bag of Words (BOW) approach. Classification algorithms, notably Random Forest and Naive Bayes, were employed, with the combination of BOW and Random Forest achieving an optimal accuracy rate of approximately 80%. Further analysis provided valuable insights into the reviews and the products they pertain to. Such insights are instrumental for e-commerce platforms like Amazon in gaining a nuanced understanding of consumer preferences and the sentiments tied to various products, thereby enhancing the precision of product recommendations within specific categories. Identifying negative feedback and areas criticized in reviews
illuminates the aspects of products that may require enhancements. Addressing these highlighted issues not only elevates the quality of the products but also significantly boosts customer satisfaction and cultivates brand loyalty.


## Reseach Design:
In order to evaluate the effectiveness of various vectorization techniques in building a machine learning model using the Amazon
customer reviews dataset, this study uses an experimental and comparative research approach. TF-IDF and Bag of Words are two
of the methods that are methodically examined in this study. The effectiveness of each method is evaluated against critical metrics, including accuracy, precision, recall, and F1 score. The objective of this comparison analysis is to identify the advantages and disadvantages of each vectorization technique, making it easier to choose the best one for model creation. Following a strict and repeatable process, the study design guarantees the extraction of important knowledge on the best
vectorization method for examining Amazon customer reviews.

## Version of the Libraries:
Our project was run on the specified libraries mention below. These are the kind of pre requisite for the project. As we had issue in handling version problem in different the project was with the environment and version given below.
- The pandas version is: 2.1.4
- The scikit-learn version is 1.2.2.
- The Numpy version is: 1.26.3
- NLTK version: 3.8.1
- Matplotlib version: 3.8.0
- Wordcloud version: 1.9.3
- Spacy version: 3.7.4

## About the Dataset:
The dataset used for this task includes 104852 rows and 15 columns with reviews of Amazon mobile electronics products provided by TensorFlow.

### Data Wrangling:
The Dataset was originally imported from TensorFlow; consequently, it wasn’t pre-handled. To clean this dataset, we used the following techniques.

### Handling Missing Value
Missing values are nearly 0.002% of this entire dataset. We decided to omit all the missing values to resolve this issue. Eliminating the missing values is not supposed to affect the nature of the examination or general presentation of the model due to the insignificant percentage of missing contents contrasted with the absolute dataset because the level of missing values is just under 0.002%. By utilizing this technique, we can approach good precision and representation of a large portion of the information without losing much data, giving us a more grounded and more steady reason for additional analysis and modelling processes.

## Text pre-processing:
### Stop Words Removal
In the text preprocessing, we wanted to remove stop words, according to our full_review column we had to set up custom stop words such as (‘One’, ‘Two’, ‘Three’, ‘Four’, ‘Five’, ‘Star’, ‘Stars’). Additionally, we did split, lower, sub and join techniques to eliminate the special characters and punctuation. Then we passed the
cleaned text to a corpus for further analysis.

### Stemming of Words
As identifying the base of the word would give more insights. All the words were stemmed and lemmatized in the text preprocessing. For the stemming we tried both the method of Stemming and Lemmatizinig. Finally, in preprocessing, we created a review_category with the values of positive equal to 1 and negative equal to 2 according to the value of review_label, 

## Implementing Spacy
In this combined NLTK and Spacy implementation for text processing, we aimed to enhance our analysis by incorporating both libraries. We focused on stopwords removal and word frequency analysis. For NLTK, we leveraged its English stopwords list and extended it with custom stopwords. Integrating Spacy into the process involved using its pre-trained English model for tokenization and text processing. While Spacy defaults to NLTK's stopwords, we ensured consistency by applying the same custom stopwords. The resulting Spacy FreqDist showcased 17,769 samples and 20,000 outcomes. This unified approach allowed us to harness the strengths of both NLTK and Spacy, providing a more comprehensive understanding of word frequencies in the dataset.
In the comparison between NLTK and Spacy, we noted distinctive performance characteristics. Despite Spacy exhibiting a longer
runtime, its robust capabilities in parsing, part-of-speech (POS) tagging, and named entity recognition (NER) make it a powerful
language model. However, recognizing that our task primarily required tokenization and word frequency analysis, we chose to streamline the process. We opted out of the additional overhead associated with parse tree generation, POS tagging, and NER in Spacy. This decision aimed at optimizing computational efficiency, as these features were unnecessary for our specific analysis.
Moreover, we acknowledged Spacy's reputation for being memory-intensive, prompting us to tailor our approach to the specific
needs of our text analysis. Incorporating stemming and lemmatization in both NLTK and Spacy further refined our text processing. In NLTK, stemming and lemmatization were performed sequentially, with the lemmatization step utilizing the output of stemming. For Spacy, we
leveraged its in-built lemmatization capabilities directly. Additionally, we utilized Spacy's general attributes, such as efficient
tokenization and language model features, contributing to a more holistic text processing pipeline. This experience served as a
valuable learning moment, underscoring the importance of selecting tools judiciously based on the specific requirements of a given task to achieve optimal performance.

## Vectorization:
Once preprocessing was done, the vectorization technique was applied to each word in full_review, corpus and for only one review to get the weight of each word. The strategies used for this task are bags of words (BOW) and TF-IDF. For each method, word cloud representation has been worked to visualize the recurrence of each word. The point is to fabricate classification models on the data by utilizing different vectorization techniques and thoroughly analyze and complete the best strategy.

### Bag of words:
The Bag of Words model is a simple representation used in Natural Language processing. In this technique, the words are addressed given their variety dismissing their punctuation and the word request. 

### TF-IDF:
TF-IDF is a result of two insights, term recurrence, and backwards archive recurrence. Term recurrence is characterized as a few times a term happens T happens in record X. Converse record recurrence is the action that lets us know how much data a word gives, or at least, whether the term is normal or interesting across all reports. It is provided by taking the logarithm of the proportion of a few reports Y in the corpus X to the quantity of the records X containing the term T.

## Modelling
After all the preprocessing step completed in the whole dataset the dataset is splitted into the percentage of 80% and 20%. In which the 80% of the data will be utilized for the training aspect of the project while the remaining 20% will be used for the testing purpose.

### Model 1: Classification Model using the Random Forest
- Random Forest algorithm works on the idea generating multiple decision tree randomly. The generation of decision tree occurs during the training process of split training dataset. During the mentioned process itself it praise the result for the final classification of the model. This is also known as ensemble method as the algorithm is grouping of multiple decision tree to form a forest which result to the final model output. This approach benefits the project in various ways such as it intensify the validity of the model and overcome one of the main complication faced in the models related to single decision tree algorithm which is overfitting.

- Scikit – learn library is utilized in this project for the model building of the Random Forest as this library helps in the combining
of various parameter and the validity provided by this library also facilitated the project. This will be constructive for the NLP application build by this model. This algorithm has been performed with both the vectorization methods such Bag of Words and TF-IDF.

### Model 2: Classification Model using the Naive Bayes
- Naive Bayes falls under the category of the supervised learning algorithm. This algorithm uses the attributes of the dataset to
predict the target column. The reason naive is as this algorithm consider all the attribute in the dataset does not have any correlation among them which is not real scenario in the real world dataset.
- To build this model also Scikit – learn libray is utilized for the primacy is mentioned in the model1 above. In this project the
authors performed the Naïve Bayes algorithm with the both the vectorization methods of Bag of Words and TF-IDF.


### Modelling Summary
These models were combined with the both vectorization method such as Bag of Word and TFIDF which explored by the developer in the initial stage of the project. Technically it is totally four models has been performed in this project.

## Results
After completion all the necessary preprocessing step which consist of managing the missing value and omission of un necessary column from the source dataset authors were able obtain the dataset with size of 2 columns with 104849 records. List of the Corpus from the dataset is designed after applying the proper NLP preprocessing techniques.
The word ‘good’ has the excessive frequency, from this it is possible to interpret the dataset which contains the review of electronic product has excessive positive response as reviews. Word cloud which is also known as tag cloud in common too. The pictorial representation of frequency distribution of word or tag in the text data is defined as word cloud. As the requirement of the project the word cloud for the dataset developers are working has been provided below for the both vectorization techniques of Bag of Words and TF-IDF. This word diagram helps find much more insight in visualized form. In which the size of each word in the word cloud depict the frequency of them in the
input or the dataset we provided. This method of visualizing the word in the cloud in ease the qualitative understanding of the words extract by vectorization method from the dataset.

## Metrics of the Models
Metrics is an evaluation in the measures of the model performance in the quantitative manner. As the project is focusing on the classification in this section will be focused in the metrices related to the classification model. In the project developers analyze all the metrices available for models formed with the different vectorization of Bag of Word and TF-IDF.
Metrices used in this project is Accuracy, Precision, Recall and F1 Score. Although ultimately these metrices depends on the
values True Positive(TP), True Negative(TN) ,False Positive(FP) and False Negative(FN). The insight we can gain from the different metrices is different.
All the metrices result will be summarized in the report below. But, the visual representation of metrices type can be done for the classification related analysis. So on the next section will look into the confusion matrix for the each type of vectorization with the two  model developed.

The table provides an overview of the performance metrics for classification algorithms (Random Forest and Naive Bayes) using different vectorization techniques (Bag of Words (BOW) and TF-IDF). Here's a summary of the results:

<table>
    <tr>
        <td>Classification Algorithms</td>
    </tr>
    <tr>
        <td>Vectorization Techniques</td>
    </tr>
    <tr>
        <td></td>
        <td>BOW Vectorizer</td>
        <td></td>
        <td></td>
        <td>TF-IDF</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>Accuracy</td>
        <td>F1 score</td>
        <td>Precision</td>
        <td>Accuracy</td>
        <td>F1 score</td>
        <td>Precision</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>0.81</td>
        <td>0.81</td>
        <td>0.81</td>
        <td>0.8</td>
        <td>0.8</td>
        <td>0.8</td>
    </tr>
    <tr>
        <td>Naive Bayes</td>
        <td>0.82</td>
        <td>0.82</td>
        <td>0.82</td>
        <td>0.79</td>
        <td>0.78</td>
        <td>0.8</td>
    </tr>
    
</table>

## Conclusions and Future Work
The choice of vectorization technique impacts model performance, with BOW generally delivering better results in this scenario. Naive Bayes appears to be a more suitable algorithm for this classification task based on the provided metrics. Further analysis and experimentation could explore additional vectorization methods and algorithms to optimize performance.
In the future, refining model performance could involve tweaking model parameters through hyperparameter tuning, considering ensemble methods for a combined model approach, and exploring deep learning techniques like recurrent neural networks. Feature engineering, or creating new relevant features, could further enhance the models. Implementing robust cross-validation and adapting the models for real-time analysis would ensure their reliability and responsiveness to new data. Fine-tuning models specifically for sentiment analysis, experimenting with additional vectorization techniques, and focusing on interpretability for
user trust are other avenues to explore. Additionally, extending the analysis to handle multiclass sentiment scenarios, integrating a user-friendly interface for presenting insights, and continuous monitoring and updating of models for evolving consumer review patterns are vital for sustained effectiveness.
