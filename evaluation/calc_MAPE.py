"""
calculate the MAE for test_data
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from joblib import dump, load
import subprocess
import pickle
from src.price import test

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
    test(X_test, y_test, model)
