import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Prepare
dataset = pd.read_table("C:/Users/Bartek/Desktop/datasets/train.tsv")
dataset["log_price"] = dataset.price.apply(lambda x:np.log(x+1))
english_stop_words = nltk.corpus.stopwords.words('english')
#nltk.download('punkt')
stemmer = PorterStemmer()
dataframe_test = pd.read_table("C:/Users/Bartek/Desktop/datasets/test.tsv")



def text_stemmer(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)  # Tokenize the text into individual words
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Perform stemming on each word  #poprawiono przetwarzanie tekstu i danych wejściowych
    return ' '.join(stemmed_tokens)  # Join the stemmed tokens back into a single text

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
    if isinstance(text, str):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in english_stop_words]
        return ' '.join(filtered_tokens)
    else:
        return text

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
dataset['brand_name'][dataset.brand_name.isnull()] = "missing"
dataset['name'][dataset.name.isnull()] = "missing"
dataset['item_condition_id'][dataset.item_condition_id.isnull()] = "missing"

dataset['category_name'] = dataset.category_name.apply(text_stemmer)


#Splitting the category name column into three levels
dataset['Tier_1'] = dataset.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>=1 else "missing")
dataset['Tier_2'] = dataset.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataset['Tier_3'] = dataset.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>2 else "missing")

#Preprocessing brand_name
dataset['brand_name'] = dataset.brand_name.apply(text_preprocessing)

#Preprocessing name
dataset['name'] = dataset.name.apply(text_preprocessing)
dataset['name'] = dataset.name.apply(remove_stop_words)

# Applying text preprocessing to item description
dataset['item_description'] = dataset.item_description.apply(decontraction)
dataset['item_description'] = dataset.item_description.apply(text_stemmer)
dataset['item_description'] = dataset.item_description.apply(remove_stop_words)
dataset['item_description'] = dataset.item_description.apply(text_preprocessing)




dataset.to_csv("C:/Users/Bartek/Desktop/Datasets/processed_train_dataset.csv")