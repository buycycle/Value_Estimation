import numpy as np
import os
import sys
import time
import pickle
from joblib import dump, load
from typing import List, Optional 

from quantile_forest import ExtraTreesQuantileRegressor
from buycycle.logger import Logger
from src.data import create_data_model
from src.price import predict_interval, predict_point_estimate
from src.driver import (
    main_query,
    main_query_dtype,
    categorical_features,
    numerical_features,
    target,
)
from src.plots import *


para = sys.argv


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true: array-like, true values.
    - y_pred: array-like, predicted values.

    Returns:
    - MAPE: float, Mean Absolute Percentage Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def cal_inside_interval(y_true, interval):
    count_within_intervals = 0
    for i, row in interval.iterrows():
        count_within_intervals += (
            (y_true.iloc[i] >= row["low"]) & (y_true.iloc[i] <= row["high"])
        ).sum()

    percentage_within_intervals = (count_within_intervals / len(y_true)) * 100
    return percentage_within_intervals


def record_model(
    note: str,
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
):
    data_dir = "data"
    sub_dir = note

    # Construct the full path
    path = os.path.join(data_dir, sub_dir)
    os.makedirs(path, exist_ok=True)

    # create model, save X_train,X_test, y_train, y_test, model, data_transform_pipeline in the folder
    create_data_model(
        path=path,
        main_query=main_query,
        main_query_dtype=main_query_dtype,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        model=ExtraTreesQuantileRegressor,
        target=target,
        test_size=0.12,
        parameters={
            "n_jobs": -1,  # Use all cores for training
        },
    )

    # get preds and interval and save
    model = load(f"{path}/model.joblib", mmap_mode=None)
    with open(f"{path}/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    quantiles = [0.05, 0.5, 0.95]
    environment = os.getenv("ENVIRONMENT")
    ab = os.getenv("AB")
    app_name = "price"
    app_version = "canary-001"
    logger = Logger.configure_logger(environment, ab, app_name, app_version)

    strategy, preds, interval, error = predict_interval(
        X_test, model, quantiles, logger
    )
    df_interval = pd.DataFrame(interval, columns=["low", "high"])
    df_interval.to_pickle(path + "/interval.pkl")
    p = pd.Series(preds, name="prediction")
    p.to_pickle(path + "/preds.pkl")

    with open(f"{path}/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    print(
        "Mean absolute percentage error: ",
        mean_absolute_percentage_error(y_test, preds),
    )
    print(
        cal_inside_interval(y_test, df_interval),
        " percentage of real price fall inside the interval",
    )
