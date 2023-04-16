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
dataframe_train, dataframe_validation = sklearn.model_selection.train_test_split(dataset, test_size=0.3, random_state=3)

dataframe_test = pd.read_table("C:/Users/Bartek/Desktop/Datasets/test.tsv")


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
dataframe_train['category_name'][dataframe_train.category_name.isnull()] = "missing"
dataframe_validation['category_name'][dataframe_validation.category_name.isnull()] = "missing"
dataframe_test['category_name'][dataframe_test.category_name.isnull()] = "missing"

dataframe_train['category_name'] = dataframe_train.category_name.apply(category_name_decontraction)
dataframe_train['category_name'] = dataframe_train.category_name.apply(category_name_preprocessing)

dataframe_validation['category_name'] = dataframe_validation.category_name.apply(category_name_decontraction)
dataframe_validation['category_name'] = dataframe_validation.category_name.apply(category_name_preprocessing)

dataframe_test['category_name'] = dataframe_test.category_name.apply(category_name_decontraction)
dataframe_test['category_name'] = dataframe_test.category_name.apply(category_name_preprocessing)

#Splitting the category name column into three levels
dataframe_train['Tier_1'] = dataframe_train.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>=1 else "missing")
dataframe_train['Tier_2'] = dataframe_train.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataframe_train['Tier_3'] = dataframe_train.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")

dataframe_validation['Tier_1'] = dataframe_validation.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>=1 else "missing")
dataframe_validation['Tier_2'] = dataframe_validation.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataframe_validation['Tier_3'] = dataframe_validation.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")
dataframe_test['Tier_1'] = dataframe_test.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>1 else "missing")
dataframe_test['Tier_2'] = dataframe_test.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
dataframe_test['Tier_3'] = dataframe_test.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")

#Preprocessing brand_name
dataframe_train['brand_name'] = dataframe_train.brand_name.apply(text_preprocessing)
dataframe_validation['brand_name'] = dataframe_train.brand_name.apply(text_preprocessing)
dataframe_test['brand_name'] = dataframe_test.brand_name.apply(text_preprocessing)

#Preprocessing name
dataframe_train['name'] = dataframe_train.name.apply(text_preprocessing)
dataframe_train['name'] = dataframe_train.name.apply(remove_stop_words)
dataframe_validation['name'] = dataframe_validation.name.apply(text_preprocessing)
dataframe_validation['name'] = dataframe_validation.name.apply(remove_stop_words)
dataframe_test['name'] = dataframe_test.name.apply(text_preprocessing)
dataframe_test['name'] = dataframe_test.name.apply(remove_stop_words)
#Removing contents of column1 from coulumn2
#def remove_name_from_brand_name(row):
#    return str(row['name']).replace(str(row['brand_name']),'')

#dataset['name'] = dataset.apply(remove_name_from_brand_name, axis=1)


# Applying text preprocessing to item description
dataframe_train['item_description'] = dataframe_train.item_description.apply(decontraction)
dataframe_train['item_description'] = dataframe_train.item_description.apply(text_preprocessing)
dataframe_train['item_description'] = dataframe_train.item_description.apply(remove_stop_words)

dataframe_validation['item_description'] = dataframe_validation.item_description.apply(decontraction)
dataframe_validation['item_description'] = dataframe_validation.item_description.apply(text_preprocessing)
dataframe_validation['item_description'] = dataframe_validation.item_description.apply(remove_stop_words)

dataframe_test['item_description'] = dataframe_test.item_description.apply(decontraction)
dataframe_test['item_description'] = dataframe_test.item_description.apply(text_preprocessing)
dataframe_test['item_description'] = dataframe_test.item_description.apply(remove_stop_words)

#Applying TfIdf to item description
tfidf_description = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
tfidf_vec_train_description = tfidf_description.fit_transform(dataframe_train.item_description)
tfidf_vec_validation_descritpion = tfidf_description.fit_transform(dataframe_validation.item_description)
tfidf_vec_test_description = tfidf_description.fit_transform(dataframe_test.item_description)
print(tfidf_vec_train_description.shape)
print(tfidf_vec_validation_descritpion.shape)
print(tfidf_vec_test_description.shape)

#Applying TfIdf to name
tfidf_name = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2), max_features=50000)
tfidf_vec_train_name = tfidf_name.fit_transform(dataframe_train.name)
tfidf_vec_validation_name = tfidf_name.fit_transform(dataframe_validation.name)
tfidf_vec_test_name = tfidf_name.fit_transform(dataframe_test.name)
print(tfidf_vec_train_name.shape)
print(tfidf_vec_validation_name.shape)
print(tfidf_vec_test_name.shape)

#Applying One Hot Encoding to item_condition_id, category_name(in different tiers), brand_name, shipping
encoded_category_tier1 = OneHotEncoder( handle_unknown='ignore')
train_encoded_category_tier1 = encoded_category_tier1.fit_transform(dataframe_train.Tier_1.values.reshape(-1,1))
validation_encoded_category_tier1 = encoded_category_tier1.fit_transform(dataframe_validation.Tier_1.values.reshape(-1,1)) #Reshaping the data to turn it into 2d array
test_encoded_category_tier1 = encoded_category_tier1.fit_transform(dataframe_test.Tier_1.values.reshape(-1,1))
print(train_encoded_category_tier1.shape)
print(validation_encoded_category_tier1.shape)
print(test_encoded_category_tier1.shape)

encoded_category_tier2 = OneHotEncoder( handle_unknown='ignore')
train_encoded_category_tier2 = encoded_category_tier2.fit_transform(dataframe_train.Tier_2.values.reshape(-1,1))
validation_encoded_category_tier2 = encoded_category_tier2.fit_transform(dataframe_validation.Tier_2.values.reshape(-1,1))
test_encoded_category_tier2 = encoded_category_tier2.fit_transform(dataframe_test.Tier_2.values.reshape(-1,1))
print(train_encoded_category_tier2.shape)
print(validation_encoded_category_tier2.shape)
print(test_encoded_category_tier2.shape)

encoded_category_tier3 = OneHotEncoder( handle_unknown='ignore')
encoded_category_tier3.fit(dataframe_test.Tier_3.values.reshape(-1,1))

train_encoded_category_tier3 = encoded_category_tier3.transform(dataframe_train.Tier_3.values.reshape(-1,1))
validation_encoded_category_tier3 = encoded_category_tier3.transform(dataframe_validation.Tier_3.values.reshape(-1,1))
test_encoded_category_tier3 = encoded_category_tier3.transform(dataframe_test.Tier_3.values.reshape(-1,1))
print(train_encoded_category_tier3.shape)
print(validation_encoded_category_tier3.shape)
print(test_encoded_category_tier3.shape)

encoded_shipping = OneHotEncoder(handle_unknown='ignore')
train_encoded_shipping = encoded_shipping.fit_transform(dataframe_train.shipping.values.reshape(-1,1))
validation_encoded_shipping = encoded_shipping.fit_transform(dataframe_validation.shipping.values.reshape(-1,1))
test_encoded_shipping = encoded_shipping.fit_transform(dataframe_test.shipping.values.reshape(-1,1))
print(train_encoded_shipping.shape)
print(validation_encoded_shipping.shape)
print(test_encoded_shipping.shape)

encoded_item_condition = OneHotEncoder(handle_unknown='ignore')
train_encoded_item_condition = encoded_item_condition.fit_transform(dataframe_train.item_condition_id.values.reshape(-1,1))
validation_encoded_item_condition = encoded_item_condition.fit_transform(dataframe_validation.item_condition_id.values.reshape(-1,1))
test_encoded_item_condition = encoded_item_condition.fit_transform(dataframe_test.item_condition_id.values.reshape(-1,1))
print(train_encoded_item_condition.shape)
print(validation_encoded_item_condition.shape)
print(test_encoded_item_condition.shape)


encoded_brand_name = OneHotEncoder(handle_unknown='ignore')
encoded_brand_name.fit(dataframe_train.brand_name.values.reshape(-1,1))
train_encoded_brand_name = encoded_brand_name.transform(dataframe_train.brand_name.values.reshape(-1,1))
validation_encoded_brand_name = encoded_brand_name.transform(dataframe_validation.brand_name.values.reshape(-1,1))
test_encoded_brand_name = encoded_brand_name.transform(dataframe_test.brand_name.values.reshape(-1,1))
print(train_encoded_brand_name.shape)
print(validation_encoded_brand_name.shape)
print(test_encoded_brand_name.shape)




final_train_vector = hstack((tfidf_vec_train_name, tfidf_vec_train_description,
                          train_encoded_category_tier1,train_encoded_category_tier2,
                            train_encoded_category_tier3,
                          train_encoded_item_condition,
                           train_encoded_brand_name, train_encoded_shipping))

final_validation_vector = hstack((tfidf_vec_validation_name, tfidf_vec_validation_descritpion,
                          validation_encoded_category_tier1,validation_encoded_category_tier2,
                            validation_encoded_category_tier3,
                          validation_encoded_item_condition,
                           validation_encoded_brand_name, validation_encoded_shipping))

final_test_vector = hstack((tfidf_vec_test_name, tfidf_vec_test_description,
                          test_encoded_category_tier1,test_encoded_category_tier2,
                            test_encoded_category_tier3,
                          test_encoded_item_condition,
                           test_encoded_brand_name, test_encoded_shipping))



print(final_train_vector.shape)
print(final_validation_vector.shape)
print(final_test_vector.shape)
def log_to_actual(log):
    return np.exp(log)-1

print("Rozpoczynam tworzenie modelu Regresji liniowej")
linear_regression = sklearn.linear_model.LinearRegression()
y_train = dataframe_train.log_price
y_validation = dataframe_validation.log_price
print("Tworzenie modelu testowego")
linear_regression.fit(final_train_vector, y_train)
print("Rozpoczynam predykcje na danych treningowych")
linear_regression_train_predictions = linear_regression.predict(final_train_vector)
print("Rozpoczynam predykcje na danych walidacyjnych")
linear_regression_validation_predictions = linear_regression.predict(final_validation_vector)


train_error = np.sqrt(mean_squared_error(y_train,linear_regression_train_predictions)) #Calculating RMSLE of Linear Regression model
print("Błąd danych treningowych:", train_error)

validation_error = np.sqrt(mean_squared_error(y_validation, linear_regression_validation_predictions))
print("Błąd danych walidacyjnych:", validation_error)

linear_regression_df = pd.DataFrame()
linear_regression_df["ID_test"] = dataframe_test.index
linear_regression_df["price"] = log_to_actual(linear_regression.predict(final_test_vector))
linear_regression_df.to_csv("C:/Users/Bartek/Desktop/test_predictions.csv")

print(linear_regression_df.head(20))
