"""test fixutres used in the tests"""

import os
import pytest

from src.data import ModelStore, read_data

from src.data import create_data_model
from quantile_forest import ExtraTreesQuantileRegressor

from src.driver import (
    target,
    categorical_features,
    numerical_features,
    main_query,
    main_query_dtype,
)

from unittest.mock import Mock, patch
from buycycle.logger import Logger


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
        months=2,
        parameters={
            "n_jobs": -1,  # Use all cores for training
        },
    )

    X_train, y_train, X_test, y_test = read_data()
    # create model store

    model_store = ModelStore()
    model_store.read_data()

    return model_store, X_train, X_test, y_train, y_test


@pytest.fixture(scope="package")
def logger_mock():
    "mock the Logger"
    # Create a mock Logger instance
    logger_mock = Mock(spec=Logger)

    return logger_mock


@pytest.fixture(scope="package")
def app_mock(logger_mock):
    "patch the model with the logger mock version and prevent threads from starting"

    with patch("buycycle.logger.Logger", return_value=logger_mock), patch(
        "src.data.ModelStore.read_data_periodically"
    ):
        # The above patches will replace the actual methods with mocks that do nothing
        from model.app import app  # Import inside the patch context to apply the mock

        yield app  # Use yield to make it a fixture
