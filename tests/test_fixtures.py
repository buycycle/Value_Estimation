"""test fixutres used in the tests"""

import os
import pytest

from flask import Flask
from flask.logging import create_logger

from src.data import ModelStore, read_data, fit_transform, create_data

from src.data import create_data_model
from quantile_forest import ExtraTreesQuantileRegressor

from src.driver import target, categorical_features, numerical_features, test_query, main_query, main_query_dtype


@pytest.fixture(scope="package")
def inputs():
    bike_id = 442745
    template_id = 41441
    msrp = 2299.0
    bike_created_at_year = 2023
    bike_created_at_month = 10
    bike_year = 2022
    sales_duration = 14
    sales_country_id = 150
    bike_type_id = 1
    bike_category_id = 2
    mileage_code = "less_than_500"
    city = "München"
    condition_code = 3
    frame_size = 52
    rider_height_min = 161.0
    rider_height_max = 169.0
    brake_type_code = "hydraulic"
    frame_material_code = "aluminum"
    shifting_code = "mechanical"
    bike_component_id = 15
    color = "#D1D5DB"
    family_model_id = 9169
    family_id = 2643
    brand_id = 173
    quality_score = 28
    is_mobile = 0
    currency_id = 2
    seller_id = 41011
    is_ebike = 0
    is_frameset = 0

    app = Flask(__name__)
    logger = create_logger(app)

    return (
        bike_id,
        template_id,
        msrp,
        bike_created_at_year,
        bike_created_at_month,
        bike_year,
        sales_duration,
        sales_country_id,
        bike_type_id,
        bike_category_id,
        mileage_code,
        city,
        condition_code,
        frame_size,
        rider_height_min,
        rider_height_max,
        brake_type_code,
        frame_material_code,
        shifting_code,
        bike_component_id,
        color,
        family_model_id,
        brand_id,
        quality_score,
        is_mobile,
        currency_id,
        seller_id,
        is_ebike,
        is_frameset,
        app,
        logger,
    )


@pytest.fixture(scope="package")
def testdata():
    # make folder data if not exists
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    _, _ = create_data(
        query=main_query, query_dtype=main_query_dtype, numerical_features=numerical_features, categorical_features=categorical_features, target=target, months=2, path="./data/"
    )
    X_train, y_train, X_test, y_test = read_data()

    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="package")
def testmodel():
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
        parameters=None,
    )

    # create model store

    model_store = ModelStore()
    model_store.read_data()

    return model_store
