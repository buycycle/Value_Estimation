"""
save the response from the fastapi model for checking the prediction/interval
testing data is from the bikes_template table
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from buycycle.logger import Logger
from src.data import get_data
from evaluate import MockApp

# Set the environment variable to indicate a test environment to skip the model selecting
os.environ["ENVIRONMENT"] = "test"

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


# prepare the testing data
df = get_data(query, query_dtype, index_col="template_id")
df = df.reset_index()

# get 10% of the data for testing
df = df.sample(frac=0.1, random_state=40)

# add condition_code and create date
df["condition_code"] = "3"

# entry point for running the evaluation
if __name__ == "__main__":
    app = MockApp()
    app.save_response(df, "data")
