import re
import sklearn
import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import nltk

# Prepare dataset
dataset = pd.read_table("C:/Users/Bartek/Desktop/Datasets/train.tsv")
english_stop_words = nltk.corpus.stopwords.words('english')

# Data preprocessing methods
def description_preprocessing(text):
    text = re.sub("[^A-Za-z0-9 ]", "", str(text))
    text = str(text).lower()
    return text
def category_name_preprocessing(text):
    text = re.sub("[^A-Za-z0-9/ ]", "", str(text))
    text = str(text).lower()
    return text

def remove_stop_words(text):
    for stopword in english_stop_words:
        stopword = ' ' + stopword + ' '
        text = str(text).replace(stopword, ' ')
        return text
def category_name_decontraction(phrase):
    phrase = re.sub(r"s", "", str(phrase))
    return phrase

def decontraction(phrase):
    # specific words
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

#Preprocessing category name column
dataset['category_name'][dataset['category_name'].isnull()] = "missing"
dataset['category_name'] = dataset['category_name'].apply(category_name_decontraction)
dataset['category_name'] = dataset['category_name'].apply(category_name_preprocessing)
#Splitting the category name column into three levels using lambda is faster than for loop
dataset['Tier_1'] = dataset['category_name'].apply(lambda x:    x.split("/")[0] if len(x.split("/"))>=1 else "missing")
dataset["Tier_2"] = dataset['category_name'].apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataset["Tier_3"] = dataset['category_name'].apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")

#Removing brand_name from name
print(dataset['Tier_1'].head(10))

# Applying text preprocessing
#dataset['item_description'] = dataset['item_description'].apply(decontraction)

#dataset['item_description'] = dataset['item_description'].apply(description_preprocessing)

#dataset['item_description'] = dataset['item_description'].apply(remove_stop_words)

#tfidf = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
#tfidf_vec = tfidf.fit_transform(dataset['item_description'])
#def preprocess_categories(text):
