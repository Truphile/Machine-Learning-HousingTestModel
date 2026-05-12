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

def split_data(housing_data, test_size, random_state):
    X = housing_data.drop(columns='median_house_value')
    y = housing_data['median_house_value']
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def build_training_pipeline(processed_data, algorithm):
    return make_pipeline(processed_data, algorithm)

splited_housing_data = split_data(housing_data, test_size=0.2, random_state=42)
splited_housing_data[0].shape

X_train = splited_housing_data[0]
X_test = splited_housing_data[1]
y_train = splited_housing_data[2]
y_test = splited_housing_data[3]


algorithms = {
    "linear_regression": LinearRegression(),
    "random_forest_regression": RandomForestRegressor(),
    "decision_tree": DecisionTreeRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "sgd_regressor": SGDRegressor(),

}
results = []
for name, algorithm in algorithms.items():
    pipeline = build_training_pipeline(clean_data(), algorithm)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_train)
    train_root_mean_square_error = root_mean_squared_error(y_train, y_pred)

    cross_validation_scores = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    results.append(
        {
            "algorithm": name,
            "train_root_mean_square_error": train_root_mean_square_error,
            # "cross_validation_scores": cross_validation_scores,
            "cross_validation_scores_mean": cross_validation_scores.mean(),
            "cross_validation_scores_std": cross_validation_scores.std(),
            "cross_validation_scores_min": cross_validation_scores.min(),
            "cross_validation_scores_max": cross_validation_scores.max()


        }
    )



