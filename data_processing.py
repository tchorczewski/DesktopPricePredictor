import re
import numpy as np
import pandas as pd
import nltk

# Prepare
dataset = pd.read_table("C:/Users/Bartek/Desktop/datasets/train.tsv")
dataset["log_price"] = dataset.price.apply(lambda x:np.log(x+1))
english_stop_words = nltk.corpus.stopwords.words('english')

dataframe_test = pd.read_table("C:/Users/Bartek/Desktop/datasets/test.tsv")


# Data preprocessing methods
def text_preprocessing(text):
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
    phrase = re.sub(r"\  "," ", str(phrase))
    return phrase

#Preprocessing category name column
dataset['category_name'][dataset.category_name.isnull()] = "missing"
dataframe_test['category_name'][dataframe_test.category_name.isnull()] = "missing"
dataset['brand_name'][dataset.brand_name.isnull()] = "missing"
dataset['name'][dataset.name.isnull()] = "missing"
dataset['item_condition_id'][dataset.item_condition_id.isnull()] = "missing"

dataset['category_name'] = dataset.category_name.apply(category_name_decontraction)
dataset['category_name'] = dataset.category_name.apply(category_name_preprocessing)

dataframe_test['category_name'] = dataframe_test.category_name.apply(category_name_decontraction)
dataframe_test['category_name'] = dataframe_test.category_name.apply(category_name_preprocessing)

#Splitting the category name column into three levels
dataset['Tier_1'] = dataset.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>=1 else "missing")
dataset['Tier_2'] = dataset.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataset['Tier_3'] = dataset.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")

dataframe_test['Tier_1'] = dataframe_test.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>1 else "missing")
dataframe_test['Tier_2'] = dataframe_test.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataframe_test['Tier_3'] = dataframe_test.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")

#Preprocessing brand_name
dataset['brand_name'] = dataset.brand_name.apply(text_preprocessing)

dataframe_test['brand_name'] = dataframe_test.brand_name.apply(text_preprocessing)

#Preprocessing name
dataset['name'] = dataset.name.apply(text_preprocessing)
dataset['name'] = dataset.name.apply(remove_stop_words)
dataframe_test['name'] = dataframe_test.name.apply(text_preprocessing)
dataframe_test['name'] = dataframe_test.name.apply(remove_stop_words)

# Applying text preprocessing to item description
dataset['item_description'] = dataset.item_description.apply(decontraction)
dataset['item_description'] = dataset.item_description.apply(text_preprocessing)
dataset['item_description'] = dataset.item_description.apply(remove_stop_words)

dataframe_test['item_description'] = dataframe_test.item_description.apply(decontraction)
dataframe_test['item_description'] = dataframe_test.item_description.apply(text_preprocessing)
dataframe_test['item_description'] = dataframe_test.item_description.apply(remove_stop_words)

dataset.to_csv("C:/Users/Bartek/Desktop/Datasets/processed_train_dataset.csv")
dataframe_test.to_csv("C:/Users/Bartek/Desktop/Datasets/processed_test_dataset.csv")