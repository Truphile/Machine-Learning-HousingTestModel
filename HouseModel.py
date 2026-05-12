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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    num_column = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    cat_column = make_pipeline(
        OneHotEncoder(handle_unknown='ignore')
    )

    processed = ColumnTransformer([
        (
            'num',
            num_column,
            make_column_selector(dtype_include=np.number)
        ),
        (
            'cat',
            cat_column,
            make_column_selector(dtype_exclude=np.number)
        )
    ])



