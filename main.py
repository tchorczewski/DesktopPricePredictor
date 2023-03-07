import re
import sklearn
import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import nltk

#Prepare dataset
dataset = pd.read_table("C:/Users/Bartek/Desktop/Datasets/train.tsv")
english_stop_words = nltk.corpus.stopwords.words('english')

#Data preprocessing methods
def text_data_preprocessing(text):
    text = re.sub("[^A-Za-z0-9 ]","",str(text))
    text = str(text).lower()
    return text
def remove_stop_words(text):
    for stopword in english_stop_words:
        stopword = ' ' + stopword + ' '
        text = str(text).replace(stopword, ' ')
        return text
def decontraction(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", str(phrase))
    phrase = re.sub(r"can\'t", "can not", str(phrase))

    # general
    phrase = re.sub(r"n\'t", " not", str(phrase))
    phrase = re.sub(r"\'re", " are", str(phrase))
    phrase = re.sub(r"\'s", " is", str(phrase))
    phrase = re.sub(r"\'d", " would", str(phrase))
    phrase = re.sub(r"\'ll", " will", str(phrase))
    phrase = re.sub(r"\'t", " not", str(phrase))
    phrase = re.sub(r"\'ve", " have", str(phrase))
    phrase = re.sub(r"\'m", " am", str(phrase))
    return phrase

#Applying text preprocessing

dataset['item_description'] = dataset['item_description'].apply(decontraction)

dataset['item_description'] = dataset['item_description'].apply(text_data_preprocessing)

dataset['item_description'] = dataset['item_description'].apply(remove_stop_words)

tfidf = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2), max_features=50000)
tfidf.fit(dataset['item_description'])
tfidf_vec = tfidf.transform(dataset['item_description'])
print(tfidf_vec.shape)

