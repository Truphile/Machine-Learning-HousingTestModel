from statistics import linear_regression
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_selector
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


df = pd.read_csv("/home/truphile/Downloads/housing.csv")
df.head()

def clean_data():
    numerical_column = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    categorical_column = make_pipeline(
        OneHotEncoder(handle_unknown='ignore'),
    )
    return ColumnTransformer([
        ('num', numerical_column, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_column, make_column_selector(dtype_exclude=np.number))
    ])
