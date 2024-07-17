Develop and implement NLP techniques for Marathi sentiment analysis.
Classify Marathi text into positive, negative, or neutral sentiments accurately.
Evaluate machine learning algorithms and linguistic analysis methods for effectiveness in Marathi sentiment analysis.
Provide insights into public sentiment towards various subjects in the Marathi-speaking community.
Enhance market strategies, customer engagement, and social listening through Marathi sentiment analysis.


*Methodology:*
Data Loading and Preprocessing:

Loading Data: The code loads the Marathi text data from a CSV file into a pandas DataFrame.
Handling Missing Values: It checks for any missing values in the dataset and drops rows containing missing values using dropna().
Text Preprocessing:
Character Removal: Non-Marathi characters are removed from the text using regular expressions (re.sub('[^\u0900-\u097F]', ' ', rvw)).
Stopword Removal: Marathi stopwords, which are common words that do not carry much meaning (e.g., articles, prepositions), are removed from the text.


Feature Extraction:

The preprocessed text data is converted into a numerical representation suitable for machine learning algorithms.
CountVectorizer from scikit-learn is used to convert the text data into a matrix of token counts. Each row represents a document (text), and each column represents a unique word (token).

Model Training:
The dataset is split into training and testing sets using train_test_split() from scikit-learn.
A Logistic Regression classifier is initialized (LogisticRegression(random_state=0, max_iter=1000)) and trained on the training data (X_trn, y_trn).


Prediction:
Sentiments are predicted for the test data (X_tst) using the trained Logistic Regression classifier.

Evaluation:
The accuracy of the model is evaluated by comparing the predicted sentiment labels (y_pred) with the actual labels from the test data (y_tst).
Accuracy score is calculated using accuracy_score() from scikit-learn.


Model Training (Multinomial Naive Bayes):
Initialize a Multinomial Naive Bayes classifier (clf) from scikit-learn.
Train the Multinomial Naive Bayes classifier (clf) on the training data (X_trn, y_trn).

Prediction and Evaluation (Multinomial Naive Bayes):
Use the trained Multinomial Naive Bayes classifier to predict sentiments for the test data (X_tst).
Evaluate the performance of the Multinomial Naive Bayes model by calculating the accuracy score, comparing predicted labels with actual labels from the test data.
