import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import requests
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from quantile_forest import ExtraTreesQuantileRegressor
from buycycle.logger import Logger

# from src.data import create_data_model
# from src.price import predict_interval, predict_point_estimate
from src.driver import (
    main_query,
    main_query_dtype,
    categorical_features,
    numerical_features,
    target,
)
from src.plot import *


def send_request(
    data: pd.DataFrame, url: str = "https://price.buycycle.com/price_interval"
):
    """
    send single request to the live server and save the result
    """
    data.replace("None", np.nan, inplace=True)
    data_dic = data.to_json(orient="records")
    headers = {
        "Content-Type": "application/json",
        "strategy": "Generic",
        "version": "canary-001",
    }
    try:
        response = requests.post(url, headers=headers, data=data_dic)
        response_data = response.json()
        price = response_data.get("price", [])
        interval = response_data.get("interval", [])
        return price, interval
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print("Failed to parse JSON from the response.")


def send_batch_request(
    df: pd.DataFrame,
    size: int = 1000,
    url: str = "https://price.buycycle.com/price_interval",
    path: str = "data",
):
    """
    send request to the live server in batch and save the result
    """
    current_date = datetime.now().date()
    # Split df into chunks of 1000 rows each
    df_chunks = np.array_split(df, np.ceil(len(df) / size))
    # List to store preds and intervals
    prices = []
    intervals = []

    for chunk in df_chunks:
        price, interval = send_request(chunk)
        prices.extend(price)
        intervals.extend(interval)

    df["predition"] = prices
    df["min"], df["max"] = zip(*intervals)
    df["deviation"] = (df["user_set_price"] - df["predition"]).round(2)
    df["deviation_rate"] = (df["deviation"] / df["predition"]).round(2)

    df.to_csv(f"{path}/deviation_{current_date}.csv", index=False)

    # Filter out rows where deviation_rate is 0 or inf before calculating the mean
    filtered_df = df[
        (df["deviation_rate"] != 0) & (df["deviation_rate"] != float("inf"))
    ]
    mean_deviation_rate = filtered_df["deviation_rate"].mean().round(2)
    print(
        f"Deviation rate between user set price and recommended price for the last 14 days: mean_deviation_rate, {mean_deviation_rate}"
    )


class MockApp:
    def __init__(self):
        # Create a mock Logger instance
        self.logger_mock = self.create_logger_mock()
        # Patch the app with the logger mock version and prevent threads from starting
        self.app = self.create_app_mock(self.logger_mock)

    def create_logger_mock(self):
        """mock the Logger"""
        logger_mock = Mock(spec=Logger)
        return logger_mock

    def create_app_mock(self, logger_mock):
        """patch the model with the logger mock version and prevent threads from starting"""
        with patch("buycycle.logger.Logger", return_value=logger_mock), patch(
            "src.data.ModelStore.read_data_periodically"
        ):
            # The above patches will replace the actual methods with mocks that do nothing
            from model.app import (
                app,
            )  # Import inside the patch context to apply the mock

            return app

    def save_response(self, df: pd.DataFrame, path: str):
        """
        send the df as request and save the df with the result(predictions and intervals)
        """
        current_date = datetime.now().date()
        client = TestClient(self.app)
        data_dict = df.to_dict(orient="records")
        response = client.post("/price_interval", json=data_dict)

        if response.status_code != 200:
            raise Exception(f"Server returned error code {response.status_code}")

        # parse the response data
        data = response.json()
        price = data.get("price")
        interval = data.get("interval")

        df["prediction"] = price
        df["min"], df["max"] = zip(*interval)
        df["ratio"] = round(df["prediction"] / df["msrp"], 2)

        df.sort_values("ratio", ascending=False, inplace=True)
        df.to_csv(f"{path}/preditions_ratio_{current_date}.csv", index=False)

        # save the file
        df.sort_values("msrp", ascending=False, inplace=True)
        df.to_csv(f"{path}/preditions_msrp_{current_date}.csv", index=False)


# para = sys.argv


# def mean_absolute_percentage_error(y_true, y_pred):
#     """
#     Calculate Mean Absolute Percentage Error (MAPE).

#     Parameters:
#     - y_true: array-like, true values.
#     - y_pred: array-like, predicted values.

#     Returns:
#     - MAPE: float, Mean Absolute Percentage Error.
#     """
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# def cal_inside_interval(y_true, interval):
#     count_within_intervals = 0
#     for i, row in interval.iterrows():
#         count_within_intervals += (
#             (y_true.iloc[i] >= row["low"]) & (y_true.iloc[i] <= row["high"])
#         ).sum()

#     percentage_within_intervals = (count_within_intervals / len(y_true)) * 100
#     return percentage_within_intervals


# def record_model(
#     note: str,
#     categorical_features: Optional[List[str]] = None,
#     numerical_features: Optional[List[str]] = None,
# ):
#     data_dir = "data"
#     sub_dir = note

#     # Construct the full path
#     path = os.path.join(data_dir, sub_dir)
#     os.makedirs(path, exist_ok=True)

#     # create model, save X_train,X_test, y_train, y_test, model, data_transform_pipeline in the folder
#     create_data_model(
#         path=path,
#         main_query=main_query,
#         main_query_dtype=main_query_dtype,
#         numerical_features=numerical_features,
#         categorical_features=categorical_features,
#         model=ExtraTreesQuantileRegressor,
#         target=target,
#         test_size=0.12,
#         parameters={
#             "n_jobs": -1,  # Use all cores for training
#         },
#     )

#     # get preds and interval and save
#     model = load(f"{path}/model.joblib", mmap_mode=None)
#     with open(f"{path}/X_test.pkl", "rb") as f:
#         X_test = pickle.load(f)

#     quantiles = [0.05, 0.5, 0.95]
#     environment = os.getenv("ENVIRONMENT")
#     ab = os.getenv("AB")
#     app_name = "price"
#     app_version = "canary-001"
#     logger = Logger.configure_logger(environment, ab, app_name, app_version)

#     strategy, preds, interval, error = predict_interval(
#         X_test, model, quantiles, logger
#     )
#     df_interval = pd.DataFrame(interval, columns=["low", "high"])
#     df_interval.to_pickle(path + "/interval.pkl")
#     p = pd.Series(preds, name="prediction")
#     p.to_pickle(path + "/preds.pkl")

#     with open(f"{path}/y_test.pkl", "rb") as f:
#         y_test = pickle.load(f)

#     print(
#         "Mean absolute percentage error: ",
#         mean_absolute_percentage_error(y_test, preds),
#     )
#     print(
#         cal_inside_interval(y_test, df_interval),
#         " percentage of real price fall inside the interval",
#     )
