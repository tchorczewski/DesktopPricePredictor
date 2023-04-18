import re
import numpy as np
import sklearn
import pandas as pd
import nltk
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack #Stackowanie matryc rzadkich pionowo
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


#SequentialFeatureSelector zrobić

# Prepare 
dataset = pd.read_csv("C:/Users/Bartek/Desktop/Datasets/processed_train_dataset.csv")
dataframe_test = pd.read_csv("C:/Users/Bartek/Desktop/Datasets/processed_test_dataset.csv")

dataframe_train, dataframe_validation = sklearn.model_selection.train_test_split(dataset, test_size=0.3, random_state=3)

#Applying TfIdf to item description
tfidf_description = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
tfidf_vec_train_description = tfidf_description.fit_transform(dataframe_train.item_description.values.astype('U'))
tfidf_vec_validation_descritpion = tfidf_description.fit_transform(dataframe_validation.item_description.values.astype('U'))
tfidf_vec_test_description = tfidf_description.fit_transform(dataframe_test.item_description.values.astype('U'))

#Applying TfIdf to name
tfidf_name = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2), max_features=50000)
tfidf_vec_train_name = tfidf_name.fit_transform(dataframe_train.name.values.astype('U'))
tfidf_vec_validation_name = tfidf_name.fit_transform(dataframe_validation.name.values.astype('U'))
tfidf_vec_test_name = tfidf_name.fit_transform(dataframe_test.name.values.astype('U'))

#Applying One Hot Encoding to item_condition_id, category_name(in different tiers), brand_name, shipping
encoded_category_tier1 = OneHotEncoder( handle_unknown='ignore')
encoded_category_tier1.fit(dataframe_train.Tier_1.values.reshape(-1,1))
train_encoded_category_tier1 = encoded_category_tier1.transform(dataframe_train.Tier_1.values.reshape(-1,1))
validation_encoded_category_tier1 = encoded_category_tier1.transform(dataframe_validation.Tier_1.values.reshape(-1,1)) #Reshaping the data to turn it into 2d array
test_encoded_category_tier1 = encoded_category_tier1.transform(dataframe_test.Tier_1.values.reshape(-1,1))

encoded_category_tier2 = OneHotEncoder( handle_unknown='ignore')
encoded_category_tier2.fit(dataframe_train.Tier_2.values.reshape(-1,1))
train_encoded_category_tier2 = encoded_category_tier2.transform(dataframe_train.Tier_2.values.reshape(-1,1))
validation_encoded_category_tier2 = encoded_category_tier2.transform(dataframe_validation.Tier_2.values.reshape(-1,1))
test_encoded_category_tier2 = encoded_category_tier2.transform(dataframe_test.Tier_2.values.reshape(-1,1))

encoded_category_tier3 = OneHotEncoder( handle_unknown='ignore')
encoded_category_tier3.fit(dataframe_train.Tier_3.values.reshape(-1,1))
train_encoded_category_tier3 = encoded_category_tier3.transform(dataframe_train.Tier_3.values.reshape(-1,1))
validation_encoded_category_tier3 = encoded_category_tier3.transform(dataframe_validation.Tier_3.values.reshape(-1,1))
test_encoded_category_tier3 = encoded_category_tier3.transform(dataframe_test.Tier_3.values.reshape(-1,1))

encoded_shipping = OneHotEncoder(handle_unknown='ignore')
encoded_shipping.fit(dataframe_train.shipping.values.reshape(-1,1))
train_encoded_shipping = encoded_shipping.transform(dataframe_train.shipping.values.reshape(-1,1))
validation_encoded_shipping = encoded_shipping.transform(dataframe_validation.shipping.values.reshape(-1,1))
test_encoded_shipping = encoded_shipping.transform(dataframe_test.shipping.values.reshape(-1,1))

encoded_item_condition = OneHotEncoder(handle_unknown='ignore')
encoded_item_condition.fit(dataframe_train.item_condition_id.values.reshape(-1,1))
train_encoded_item_condition = encoded_item_condition.transform(dataframe_train.item_condition_id.values.reshape(-1,1))
validation_encoded_item_condition = encoded_item_condition.transform(dataframe_validation.item_condition_id.values.reshape(-1,1))
test_encoded_item_condition = encoded_item_condition.transform(dataframe_test.item_condition_id.values.reshape(-1,1))

encoded_brand_name = OneHotEncoder(handle_unknown='ignore')
encoded_brand_name.fit(dataframe_train.brand_name.values.reshape(-1,1))
train_encoded_brand_name = encoded_brand_name.transform(dataframe_train.brand_name.values.reshape(-1,1))
validation_encoded_brand_name = encoded_brand_name.transform(dataframe_validation.brand_name.values.reshape(-1,1))
test_encoded_brand_name = encoded_brand_name.transform(dataframe_test.brand_name.values.reshape(-1,1))

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
                            test_encoded_category_tier3, test_encoded_item_condition,
                           test_encoded_brand_name, test_encoded_shipping))

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

r2train = r2_score(y_train, linear_regression_train_predictions)
print("Wynik r2:", r2train)

validation_error = np.sqrt(mean_squared_error(y_validation, linear_regression_validation_predictions))
print("Błąd danych walidacyjnych:", validation_error)
print("R2 Validation:", r2_score(y_validation, linear_regression_validation_predictions))

linear_regression_df = pd.DataFrame()
linear_regression_df["ID_test"] = dataframe_test.index
linear_regression_df["price"] = log_to_actual(linear_regression.predict(final_test_vector))
print("R2 Score of test:")
#linear_regression_df.to_csv("C:/Users/Bartek/Desktop/test_predictions.csv")

#Model drzewa decyzyjnego - regresyjnego
#decision_tree_regressor = DecisionTreeRegressor()
#decision_tree_regressor.fit(final_train_vector, y_train)
#print("Tworzenie modelu testowego")
#train_prediction_regression = decision_tree_regressor.predict(final_train_vector, y_train)
#print("Tworzenie modelu walidacyjnego")
#validation_prediction_regression = decision_tree_regressor.predict(final_validation_vector, y_validation)

#regression_tree_train_error = np.sqrt(mean_squared_error(y_train,train_prediction_regression))
#print("Błąd modelu treningowego:", regression_tree_train_error)

#regression_tree_validation_error = np.sqrt(mean_squared_error(y_train,validation_prediction_regression))
#print("Błąd modelu treningowego:", regression_tree_validation_error)

#tree_regression_df = pd.DataFrame()
#tree_regression_df["Id_test"] = dataframe_test.index
#tree_regression_df["price"] = log_to_actual(decision_tree_regressor.predict(final_test_vector))
#print(tree_regression_df.head(20))