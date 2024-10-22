# Tweet Classification for Disaster Prediction

This repository contains code for a project aimed at classifying tweets to determine if they relate to real-life disasters. Using a dataset from Kaggle, the project explores different machine learning approaches, including Logistic Regression and BERT-based models, to classify tweets accurately.



## Overview

This project aims to predict whether a tweet is related to a real-life disaster using machine learning techniques. The dataset used for this classification task is sourced from Kaggle and contains various features such as tweet ID, keyword, location, and the tweet content itself.

## Dataset

The dataset includes the following columns:
- **id**: Unique identifier for each tweet.
- **keyword**: Specific keyword associated with the tweet.
- **location**: The location from which the tweet was posted.
- **tweet**: The text content of the tweet.

## Data Cleaning

Before training the models, the dataset underwent a series of preprocessing steps using NLP libraries. The cleaning processes included:
- Removing punctuation.
- Stemming the text.
- Removing emojis.
- Eliminating irrelevant information from the tweets.

## Models

Two machine learning models were created and evaluated for this project:

1. **Logistic Regression**: A traditional machine learning approach used for binary classification tasks.
2. **BERT (Bidirectional Encoder Representations from Transformers)**: A state-of-the-art deep learning model designed for NLP tasks.

I compared the performance of both models to assess their effectiveness in classifying tweets.
