import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack  # Stackowanie matryc rzadkich pionowo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

def log_to_actual(log):
    return np.exp(log) - 1
now = datetime.now()
current_time = now.strftime("%H:%M:%S")

print("Current Time =", current_time)
# Prepare
dataset = pd.read_csv("C:/Users/Bartek/Desktop/Datasets/processed_train_dataset.csv")

# Applying TfIdf to item description
tfidf_description = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
tfidf_vec_description = tfidf_description.fit_transform(dataset.item_description.values.astype('U'))
print(tfidf_vec_description.shape)

# Applying TfIdf to name
tfidf_name = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
tfidf_vec_name = tfidf_name.fit_transform(dataset.name.values.astype('U'))

encoded_category_tier1 = OneHotEncoder(handle_unknown='ignore')
encoded_category_tier1.fit(dataset.Tier_1.values.reshape(-1, 1))
train_encoded_tier1 = encoded_category_tier1.transform(dataset.Tier_1.values.reshape(-1, 1))

encoded_category_tier2 = OneHotEncoder(handle_unknown='ignore')
encoded_category_tier2.fit(dataset.Tier_2.values.reshape(-1, 1))
train_encoded_tier2 = encoded_category_tier2.transform(dataset.Tier_2.values.reshape(-1, 1))

encoded_category_tier3 = OneHotEncoder(handle_unknown='ignore')
encoded_category_tier3.fit(dataset.Tier_3.values.reshape(-1, 1))
train_encoded_tier3 = encoded_category_tier3.transform(dataset.Tier_3.values.reshape(-1, 1))

encoded_shipping = OneHotEncoder(handle_unknown='ignore')
encoded_shipping.fit(dataset.shipping.values.reshape(-1, 1))
train_encoded_shipping = encoded_shipping.transform(dataset.shipping.values.reshape(-1, 1))

encoded_item_condition = OneHotEncoder(handle_unknown='ignore')
encoded_item_condition.fit(dataset.item_condition_id.values.reshape(-1, 1))
train_encoded_item_condition = encoded_item_condition.transform(dataset.item_condition_id.values.reshape(-1, 1))
print(train_encoded_item_condition.shape)

encoded_brand_name = OneHotEncoder(handle_unknown='ignore')
encoded_brand_name.fit(dataset.brand_name.values.reshape(-1, 1))
train_encoded_brand_name = encoded_brand_name.transform(dataset.brand_name.values.reshape(-1, 1))
print(train_encoded_brand_name.shape)

dataset_vector = hstack(
    (tfidf_vec_description, tfidf_vec_name, train_encoded_tier1, train_encoded_tier2, train_encoded_tier3,
     train_encoded_shipping,
     train_encoded_item_condition, train_encoded_brand_name))

linear_regression = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=-1)

y = dataset['log_price']

vector_train, vector_validation, y_train, y_validation = sklearn.model_selection.train_test_split(dataset_vector, y, test_size=0.3,
                                                                           random_state=3)

print("Rozpoczynam tworzenie modelu Regresji liniowej")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

print("Tworzenie modelu testowego")
linear_regression.fit(vector_train, y_train)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
print("Rozpoczynam predykcje na danych treningowych")
linear_regression_train_predictions = linear_regression.predict(vector_train)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

train_error = np.sqrt(
    mean_squared_error(y_train, linear_regression_train_predictions))  # Calculating RMSLE of Linear Regression model
print("Błąd danych treningowych:", train_error)

r2train = r2_score(y_train, linear_regression_train_predictions)
print("Wynik r2:", r2train)

linear_regression_validation_predictions = linear_regression.predict(vector_validation)
r2train = r2_score(y_validation, linear_regression_validation_predictions)
print("Wynik r2:", r2train)
validation_error = np.sqrt(
    mean_squared_error(y_validation, linear_regression_validation_predictions))  # Calculating RMSLE of Linear Regression model
print("Błąd danych walidacyjnych:", validation_error)
