"""
evaluatte the model by comparing the user setup deviation for past month
take the data from bikes table for the last 14 days, 
compare the prediction and the user setup price
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data import get_data
from src.driver import generate_query, main_query_dtype
from evaluation.evaluate import send_batch_request, send_request

path = "data"

# main_query_selection
selection = {
    "bikes": {
        "id": {"alias": "id", "datatype": pd.Int64Dtype()},
        "msrp": {"alias": "msrp", "datatype": pd.Float64Dtype()},
        "price": {"alias": "user_set_price", "datatype": pd.Float64Dtype()},
        "common_price": {"alias": "common_price", "datatype": pd.Float64Dtype()},
        "created_at": {"alias": "created_at", "datatype": str},
        "status": {"alias": "status", "datatype": str},
    },
    "bike_less_used_columns": {
        "price_version": {"alias": "price_version", "datatype": str},
        "interval_min": {"alias": "interval_min", "datatype": pd.Float64Dtype()},
        "interval_max": {"alias": "interval_max", "datatype": pd.Float64Dtype()},
    },
}

joins = [
    {
        "type": "join",
        "table1": "bikes",
        "table2": "bike_less_used_columns",
        "t1Column": "id",
        "t2Column": "bike_id",
    },
]

# Calculate the date 14 days ago from today
days = 14
days_ago = datetime.now() - timedelta(days=days)

# Format the date in a way that matches your database's date format, e.g., 'YYYY-MM-DD'
formatted_date = days_ago.strftime("%Y-%m-%d")

# Modify the where_clause to filter data from the last 1 month
where_clause = f"""
WHERE bikes.created_at >= '{formatted_date}'
AND bike_less_used_columns.price_version = "stable-002-highprice"
"""

order_by_clause = """
ORDER BY bikes.created_at DESC
"""

query, dtype = generate_query(
    selection, joins=joins, where_clause=where_clause, order_by_clause=order_by_clause
)

df = get_data(query, dtype)
df = df.reset_index()

# remove out the common_price und user_set_price is 0 or Nan
df = df[(df["common_price"].notnull() & ~df["common_price"].eq(0)) & 
        (df["user_set_price"].notnull() & ~df["user_set_price"].eq(0))]

df["deviation"] = (df["user_set_price"] - df["common_price"]).round(2)
df["deviation_rate"] = (df["deviation"] / df["user_set_price"]).round(2)
mean_deviation_rate = df["deviation_rate"].mean().round(2)
print(
    f"Deviation rate between user set price and recommended price for the last {days} days with {df.shape} rows of data: mean_deviation_rate, {mean_deviation_rate}"
)

df["inside_interval"] = (
    (df["user_set_price"] >= df["interval_min"])
    & (df["user_set_price"] <= df["interval_max"])
).astype(int)
average_inside_interval = df["inside_interval"].mean().round(2)
print(f"{average_inside_interval} of price falls inside interval")

today_date = datetime.now().strftime("%Y-%m-%d")
df.to_csv(f"data/setup_deviation_{today_date}.csv", index=False)