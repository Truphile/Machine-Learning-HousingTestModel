import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import joblib


def train_housing_model(
    housing,
    target_col='median_house_value',
    test_size=0.2,
    random_state=42
):
    housing['ocean_proximity'] = (
        housing['ocean_proximity']
        .astype('category')
    )

    housing['total_bedrooms'] = (
        housing['total_bedrooms']
        .fillna(housing['total_bedrooms'].median())
    )


    X = housing.drop(columns=target_col)
    y = housing[target_col]

