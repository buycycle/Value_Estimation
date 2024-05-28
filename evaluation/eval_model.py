"""
save the response from the fastapi model for checking the prediction/interval
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from buycycle.logger import Logger
from src.data import get_data

# define the test data
query = """
            SELECT 
                tb1.id as template_id, 
                tb1.msrp, 
                tb1.year as bike_year, 
                tb1.bike_type_id, 
                tb1.bike_category_id, 
                tb1.brake_type_code, 
                tb1.frame_material_code, 
                tb1.shifting_code, 
                tb1.bike_component_id, 
                tb1.family_model_id, 
                tb1.family_id, 
                tb1.brand_id,
                COALESCE(tb2.is_ebike, 0) as is_ebike, 
                COALESCE(tb2.is_frameset, 0) as is_frameset
            FROM bike_templates tb1
            LEFT JOIN bike_template_additional_infos tb2  ON tb1.id = tb2.bike_template_id
            WHERE tb1.msrp IS NOT NULL
        """

query_dtype = {
    "template_id": pd.Int64Dtype(),
    "msrp": pd.Float64Dtype(),
    "bike_year": pd.Int64Dtype(),
    "bike_type_id": pd.Int64Dtype(),
    "bike_category_id": pd.Int64Dtype(),
    "brake_type_code": str,
    "frame_material_code": str,
    "shifting_code": str,
    "bike_component_id": pd.Int64Dtype(),
    "family_model_id": pd.Int64Dtype(),
    "family_id": pd.Int64Dtype(),
    "brand_id": pd.Int64Dtype(),
    "is_ebike": pd.Int64Dtype(),
    "is_frameset": pd.Int64Dtype(),
}

# get the test data
df = get_data(query, query_dtype, index_col="template_id")
df = df.reset_index()

# get 10% of the data for testing
df = df.sample(frac=0.1, random_state=1)

# add condition_code and create date
df["condition_code"] = "3"

# transfer dataframe to dictionary
data_dict = df.to_dict(orient="records")


def create_logger_mock():
    """mock the Logger"""
    logger_mock = Mock(spec=Logger)

    return logger_mock


def create_app_mock(logger_mock):
    """patch the model with the logger mock version and prevent threads from starting"""
    with patch("buycycle.logger.Logger", return_value=logger_mock), patch(
        "src.data.ModelStore.read_data_periodically"
    ):
        # The above patches will replace the actual methods with mocks that do nothing
        from model.app import app  # Import inside the patch context to apply the mock

        return app


def evaluation_model(path):
    """test time and len of return for all strategies of the fastapi app"""
    logger_mock = create_logger_mock()
    app_mock = create_app_mock(logger_mock)
    print(app_mock)

    client = TestClient(app_mock)
    request = data_dict
    response = client.post("/price_interval", json=request)

    if response.status_code != 200:
        raise Exception(f"Server returned error code {response.status_code}")
    # parse the response data
    data = response.json()
    price = data.get("price")
    interval = data.get("interval")

    global df
    # save the result and sort the result
    df["prediction"] = price
    new_df = pd.DataFrame(interval, columns=["min", "max"])
    df = df.reset_index(drop=True).join(new_df.reset_index(drop=True))
    df.sort_values("msrp", ascending=False, inplace=True)

    # save the file
    df.to_csv(path, index=False)


# entry point for running the evaluation
if __name__ == "__main__":
    evaluation_model("data/prediction.csv")
