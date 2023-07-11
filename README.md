# Amazon Fine Food Reviews Analysis

This repository contains code for analyzing and predicting sentiment from Amazon Fine Food Reviews dataset. The code is implemented in Python and utilizes various natural language processing (NLP) techniques and machine learning algorithms.

## Dataset

The dataset used for this analysis is the Amazon Fine Food Reviews dataset, which contains reviews of food products from Amazon. The dataset consists of reviews and their corresponding scores (ratings) given by users. The goal is to predict whether a review is positive or negative based on the text content.

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews). After downloading the dataset, extract the contents of the zip file to the same directory as the code files.

## Prerequisites

Make sure you have the following packages installed in your Python environment:

- `kaggle`
- `numpy`
- `pandas`
- `nltk`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `gensim`

You can install these packages using `pip`. For example, `pip install numpy`.

## Code Structure

The code is divided into several sections, each focusing on a specific task or technique. Here's an overview of the sections:

1. Data Loading and Preprocessing: The code begins by importing the required libraries and loading the dataset from an SQLite database. It then performs data cleaning and preprocessing steps such as removing HTML tags, punctuation, and stopwords, and performs stemming using the Porter stemmer.

2. Exploratory Data Analysis: This section explores the dataset by visualizing the distribution of positive and negative reviews, analyzing the most common positive and negative words, and examining the distribution of review lengths.

3. Bag of Words (BoW) Representation: The code uses the CountVectorizer from scikit-learn to convert the preprocessed text data into a numerical representation known as the Bag of Words model. It creates a sparse matrix of word counts for each review.

4. TF-IDF Representation: The code utilizes the TfidfVectorizer from scikit-learn to transform the BoW representation into TF-IDF (Term Frequency-Inverse Document Frequency) representation. This representation assigns weights to words based on their importance in the document and across the corpus.

5. Word2Vec Representation: The code demonstrates the use of pre-trained Word2Vec embeddings trained on the Google News dataset. It also shows how to train a Word2Vec model from scratch using the dataset.

6. Average Word2Vec and TF-IDF Weighted Word2Vec: This section computes average Word2Vec and TF-IDF weighted Word2Vec representations for each review. These representations capture the semantic meaning of words and provide dense vector representations.

## Results and Model Evaluation

The code presented here focuses on data preprocessing and feature extraction techniques. To build a predictive model using these features, you can apply various machine learning algorithms such as logistic regression, decision trees, random forests, or support vector machines. Perform model evaluation using appropriate metrics such as accuracy, precision, recall, and F1-score.

## Conclusion

This code repository provides a comprehensive analysis of the Amazon Fine Food Reviews dataset. It covers data preprocessing, exploratory data analysis, and different text representation techniques such as Bag of Words, TF-IDF, and Word2Vec. You can leverage this code to gain insights from the dataset and build predictive models for sentiment analysis.

Feel free to customize the code according to your requirements and experiment with different models and techniques. Enjoy exploring the world of natural language processing and machine learning!
