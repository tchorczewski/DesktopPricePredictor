import re

import numpy as np
import sklearn
import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack #Stackowanie matryc rzadkich pionowo
from sklearn.metrics import mean_squared_error

# Prepare dataset
dataset = pd.read_table("C:/Users/Bartek/Desktop/Datasets/train.tsv")
dataset["log_price"] = dataset.price.apply(lambda x:np.log(x+1))
english_stop_words = nltk.corpus.stopwords.words('english')

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
    return phrase

#Preprocessing category name column
dataset['category_name'][dataset.category_name.isnull()] = "missing"
dataset['category_name'] = dataset.category_name.apply(category_name_decontraction)
dataset['category_name'] = dataset.category_name.apply(category_name_preprocessing)
#Splitting the category name column into three levels
dataset['Tier_1'] = dataset.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>=1 else "missing")
dataset["Tier_2"] = dataset.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataset["Tier_3"] = dataset.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")

#Preprocessing brand_name
dataset['brand_name'] = dataset.brand_name.apply(text_preprocessing)

#Preprocessing name
dataset['name'] = dataset.name.apply(text_preprocessing)
dataset['name'] = dataset.name.apply(remove_stop_words)
#Removing contents of column1 from coulumn2
#def remove_name_from_brand_name(row):
#    return str(row['name']).replace(str(row['brand_name']),'')

#dataset['name'] = dataset.apply(remove_name_from_brand_name, axis=1)


# Applying text preprocessing to item description
dataset['item_description'] = dataset.item_description.apply(decontraction)

dataset['item_description'] = dataset.item_description.apply(text_preprocessing)

dataset['item_description'] = dataset.item_description.apply(remove_stop_words)

#Applying TfIdf to item description
tfidf_description = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
tfidf_vec_description = tfidf_description.fit_transform(dataset.item_description)


#Applying TfIdf to name
tfidf_name = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2), max_features=50000)
tfidf_vec_name = tfidf_name.fit_transform(dataset.name)

#Applying One Hot Encoding to item_condition_id, category_name(in different tiers), brand_name, shipping
encoded_category_tier1 = OneHotEncoder( handle_unknown='ignore')
encoded_category_tier1 = encoded_category_tier1.fit_transform(dataset.Tier_1.values.reshape(-1,1))  #Reshaping the data to turn it into 2d array

encoded_category_tier2 = OneHotEncoder( handle_unknown='ignore')
encoded_category_tier2 = encoded_category_tier2.fit_transform(dataset.Tier_2.values.reshape(-1,1))

encoded_category_tier3 = OneHotEncoder( handle_unknown='ignore')
encoded_category_tier3 = encoded_category_tier3.fit_transform(dataset.Tier_3.values.reshape(-1,1))

encoded_shipping = OneHotEncoder()
encoded_shipping = encoded_shipping.fit_transform(dataset.shipping.values.reshape(-1,1))

encoded_item_condition = OneHotEncoder()
encoded_item_condition = encoded_item_condition.fit_transform(dataset.item_condition_id.values.reshape(-1,1))

encoded_brand_name = OneHotEncoder()
encoded_brand_name = encoded_brand_name.fit_transform(dataset.brand_name.values.reshape(-1,1))


final_vector = hstack((tfidf_vec_name, tfidf_vec_description,
                          encoded_category_tier1,encoded_category_tier2, encoded_category_tier3,
                          encoded_item_condition,
                           encoded_brand_name, encoded_shipping))

def log_to_actual(log):
    return np.exp(log)-1

print("Rozpoczynam tworzenie modelu Regresji liniowej")
linear_regression = sklearn.linear_model.LinearRegression()
y_train = dataset.log_price
linear_regression.fit(final_vector, y_train)
print("Rozpoczynam predykcje")
linear_regression_predictions = linear_regression.predict(final_vector)

train_error = np.sqrt(mean_squared_error(y_train,linear_regression_predictions)) #Calculating RMSLE of Linear Regression model
print("Błąd danych treningowych:", train_error)

linear_regression_df = pd.DataFrame()
linear_regression_df["ID_test"] = dataset.index
linear_regression_df["price"] = log_to_actual(linear_regression.predict(final_vector))
linear_regression_df.to_csv("C:/Users/Bartek/Desktop/test_predictions.csv")

print(linear_regression_df.head(20))
