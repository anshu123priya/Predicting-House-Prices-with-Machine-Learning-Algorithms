from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



# divide data into predictor and target variables
X = train.drop('SalePrice', axis=1)
Y = train.SalePrice


predictor_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']
train_X = train_X[predictor_cols]

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

# one-hot encoding categorical variables for analysis
onehot_train_X = pd.get_dummies(train_X)
onehot_test_X = pd.get_dummies(test_X)
train_X, test_X = onehot_train_X.align(onehot_test_X, join='left', axis=1)

# impute missing values with the column's mean value
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# define the model_1
my_model_1 = LinearRegression()
my_model_1.fit(train_X, train_y)
predictions = my_model_1.predict(test_X)
print("Mean Absolute Error 1 : " + str(mean_absolute_error(predictions, test_y)))

# define the model_2
my_model_2 = RandomForestRegressor()
my_model_2.fit(train_X,train_y)
predictions = my_model_2.predict(test_X)
print("Mean Absolute Error 2: " + str(mean_absolute_error(predictions, test_y)))


# define the model_3
my_model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model_3.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model_3.predict(test_X)
print("Mean Absolute Error 3: " + str(mean_absolute_error(predictions, test_y)))


#final testing on real test data
test_real = test
test_X = test_real[predictor_cols]
predicted_prices = my_model.predict(test_X)
print(predicted_prices)


