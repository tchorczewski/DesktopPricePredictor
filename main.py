import datetime
import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_halving_search_cv # Enabling experimental feature
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack  # Stacking sparse matrices vertically
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import TruncatedSVD

params_dt = {'max_depth': [10, 50, 100, 200]}
params_lr = {'fit_intercept': [True, False], 'n_jobs':[1,2,-1]}

dataset = pd.read_csv("C:/Users/Bartek/Desktop/Datasets/processed_train_dataset.csv")

y = dataset['log_price']
start_time = datetime.datetime.now()
encoded_brand_name = OneHotEncoder(handle_unknown='ignore')
encoded_brand_name.fit(dataset.brand_name.values.reshape(-1, 1))
train_encoded_brand_name = encoded_brand_name.transform(dataset.brand_name.values.reshape(-1, 1))

# Selecting the best features from brand name column
#svd = TruncatedSVD(n_components=100)
#sparse_feature_transformed = svd.fit_transform(train_encoded_brand_name)
#print(sparse_feature_transformed.shape)

def log_to_actual(log):
    return np.exp(log) - 1

# Applying TfIdf to item description
tfidf_description = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
tfidf_vec_description = tfidf_description.fit_transform(dataset.item_description.values.astype('U'))
print(tfidf_vec_description.shape)

# Applying TfIdf to name
tfidf_name = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
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

dataset_vector = hstack(
    (tfidf_vec_description, tfidf_vec_name, train_encoded_tier1, train_encoded_tier2, train_encoded_tier3,
     train_encoded_shipping,
     train_encoded_item_condition, train_encoded_brand_name))
print(dataset_vector.shape)

vector_train, vector_validation, y_train, y_validation = sklearn.model_selection.train_test_split(dataset_vector, y,
                                                                                                  test_size=0.3,
                                                                                                  random_state=3)

print(vector_train.shape)
print(vector_validation.shape)
#Getting the best hyperparameters of regression tree
#print("Hiperparametry drzewa decyzyjnego")
#dt = DecisionTreeRegressor(criterion="squared_error", max_features="auto")
#grid_dt = HalvingGridSearchCV(dt, param_grid=params, cv=3, return_train_score=True, scoring='neg_mean_squared_error', n_jobs=-1)
#grid_dt.fit(vector_train, y_train)
#print("Best Hyperparameters:", grid_dt.best_params_)

#Selecting the hyperparameters of linear regression
#print("Hiperparametry regresji liniowej")
#lr = LinearRegression()
#grid_lr = GridSearchCV(lr, param_grid=params_lr, return_train_score=True, scoring='neg_mean_squared_error', n_jobs=-1)
#grid_lr.fit(vector_train, y_train)
#print("Best Hyperparameters:", grid_lr.best_params_)

end_time = datetime.datetime.now()
runtime = end_time - start_time
total_seconds = runtime.total_seconds()
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

# Print the time
print("Czas wykonania tworzenia macierzy rzadkiej: " + formatted_time)

start_time = datetime.datetime.now()
#Linear regression model
print("Rozpoczynam tworzenie modelu Regresji liniowej")
linear_regression = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=-1)
print("Tworzenie modelu testowego")
linear_regression.fit(vector_train, y_train)
print("Rozpoczynam predykcje na danych treningowych")
linear_regression_train_predictions = linear_regression.predict(vector_train)
print("Rozpoczynam predykcje na:"
      " danych walidacyjnych")
linear_regression_validation_predictions = linear_regression.predict(vector_validation)

train_error = np.sqrt(
    mean_squared_error(y_train, linear_regression_train_predictions))  # Calculating RMSLE of Linear Regression model
print("Błąd danych treningowych dla modelu Regresji Liniowej:", train_error)

r2train = r2_score(y_train, linear_regression_train_predictions)
print("Wynik r2 dla modelu Regresji Liniowej: ", r2train) # Calculating R^2 of Linear Regression

validation_error = np.sqrt(mean_squared_error(y_validation, linear_regression_validation_predictions))
print("Błąd danych walidacyjnych dla modelu Regresji Liniowej:", validation_error)
print("R2 Validation dla modelu Regresji Liniowej:", r2_score(y_validation, linear_regression_validation_predictions))
end_time = datetime.datetime.now()
runtime = end_time - start_time
total_seconds = runtime.total_seconds()

hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
# Print the score
print("Czas wykonania trenowania i predykcji dla modelu regresji liniowej: " + formatted_time)

start_time = datetime.datetime.now()
#Regression tree model
print("Rozpoczynam pracę nad modelem drzewa regresyjnego")
dt = DecisionTreeRegressor(max_depth=30, max_features='auto')
dt.fit(vector_train, y_train)
print("Rozpoczynam predykcje na danych treningowych")
dt_train_predictions = dt.predict(vector_train)
print("Rozpoczynam predykcje na danych walidacyjnych")
dt_validation_predictions = dt.predict(vector_validation)

train_error_dt = np.sqrt(mean_squared_error(y_train, dt_train_predictions))  # Calculating RMSLE of Regression tree
print("Błąd danych treningowych dla modelu Drzewa Regresyjnego:", train_error_dt)
r2train_dt = r2_score(y_train, dt_train_predictions) # Calculating R^2 of Regression tree
print("Wynik r2:", r2train_dt)

validation_error_dt = np.sqrt(mean_squared_error(y_validation, dt_validation_predictions))
print("Błąd danych walidacyjnych dla modelu Drzewa Regresyjnego:", validation_error_dt)
print("R2 Validation:", r2_score(y_validation, dt_validation_predictions))
end_time = datetime.datetime.now()
runtime = end_time - start_time
total_seconds = runtime.total_seconds()

hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
formatted_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
# Wydruk wyniku
print("Czas wykonania trenowania i predykcji dla modelu drzewa regresyjnego: " + formatted_time)