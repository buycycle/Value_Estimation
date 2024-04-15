"""test fixutres used in the tests"""

import os
import pytest

from src.data import ModelStore, read_data, fit_transform, create_data

from src.data import create_data_model
from quantile_forest import ExtraTreesQuantileRegressor

from src.driver import target, categorical_features, numerical_features, test_query, main_query, main_query_dtype



@pytest.fixture(scope="package")
def testdata():
    # make folder data if not exists
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    create_data_model(
        path="./data/",
        main_query=main_query,
        main_query_dtype=main_query_dtype,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        model=ExtraTreesQuantileRegressor,
        target=target,
        months=22,
        parameters = {
            'n_jobs': -1,                 # Use all cores for training
        },
    )

    X_train, y_train, X_test, y_test = read_data()
    # create model store

    model_store = ModelStore()
    model_store.read_data()
    
    return model_store, X_train, X_test, y_train, y_test
