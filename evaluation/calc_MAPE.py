"""
calculate the MAE for currently saved model with test_data
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from joblib import dump, load
import subprocess
import pickle
from src.price import test
from src.driver import msrp_min, msrp_max

path = "data"

files = [
    "X_test.pkl",
    "y_test.pkl",
    "model.joblib",
]

for file in files:
    if not os.path.exists(f"{path}/{file}"):
        subprocess.run(["python", "create_data.py"], check=True)

# Load data, model and pipeline from create_data.py
with open(f"{path}/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(f"{path}/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)
model = load(f"{path}/model.joblib", mmap_mode=None)


if __name__ == "__main__":
    # Choose the data inside the msrp range
    valid_indices = X_test[
        (X_test["msrp"] <= msrp_max) & (X_test["msrp"] >= msrp_min)
    ].index

    # Use the valid indices to filter both X_test and y_test
    X_test_filtered = X_test.loc[valid_indices]
    y_test_filtered = y_test.loc[valid_indices]

    test(X_test_filtered, y_test_filtered, model)
