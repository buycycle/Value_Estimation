import sys
import time
from src.driver import (
    main_query,
    main_query_dtype,
    categorical_features,
    numerical_features,
    target,
)

from src.data import create_data_model
from quantile_forest import ExtraTreesQuantileRegressor

# if there is a command line argument, use it as path, else use './data/'
path = sys.argv[1] if len(sys.argv) > 1 else "./data/"

create_data_model(
    path=path,
    main_query=main_query,
    main_query_dtype=main_query_dtype,
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    model=ExtraTreesQuantileRegressor,
    target=target,
    months=18,
    parameters=None,
)

print("created_data_model")

# sleep for 10 seconds to make sure the data saved
time.sleep(10)
