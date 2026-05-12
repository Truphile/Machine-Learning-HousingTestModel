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

    linear_reg = make_pipeline(
        processed,
        LinearRegression()
    )

    linear_reg.fit(X_train, y_train)

    linear_pred = linear_reg.predict(X_train)

    # RMSE
    linear_rmse = root_mean_squared_error(
        y_train,
        linear_pred
    )

    linear_cv = -cross_val_score(
        linear_reg,
        X_train,
        y_train,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )

    forest_model = make_pipeline(
        processed,
        RandomForestRegressor()
    )

    forest_model.fit(X_train, y_train)

    forest_pred = forest_model.predict(X_train)

    forest_rmse = root_mean_squared_error(
        y_train,
        forest_pred
    )

    test_pred = linear_reg.predict(X_test)

    test_rmse = root_mean_squared_error(
        y_test,
        test_pred
    )

    joblib.dump(
        linear_reg,
        "r_model.pkl"
    )

    return {
        "linear_rmse": linear_rmse,
        "linear_cv_mean": linear_cv.mean(),
        "forest_rmse": forest_rmse,
        "test_rmse": test_rmse,
        "model_saved": True
    }





