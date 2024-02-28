"""
 test collaborative recommendation provision realted functions
"""

import numpy as np
import time
import random

from tests.test_fixtures import testdata
from src.price import predict_interval


def test_mean_error(testdata, mean_error_limit=0.4, quantiles=[0.05, 0.5, 0.95]):
    """Test that the mean percentage error of the test data is below a certain limit."""
    model_store, X_train, X_test, y_train, y_test = testdata
    strategy, preds, intervals, error = predict_interval(X_test, model_store.regressor, quantiles, logger=None)

    percentage_errors = np.abs((y_test - preds) / y_test)
    mean_percentage_error = np.mean(percentage_errors)

    assert mean_percentage_error < mean_error_limit, f"Mean percentage error {mean_percentage_error} of test data is not below {mean_error_limit}"


def test_in_prediction_interval(testdata, interval_coverage_limit=0.7, quantiles=[0.05, 0.5, 0.95]):
    """
    Test that the sales price is between the low and high prediction interval for a certain percentage of cases.
    """
    model_store, X_train, X_test, y_train, y_test = testdata

    # and returns a strategy, predictions, intervals, and error
    strategy, preds, intervals, error = predict_interval(X_test, model_store.regressor, quantiles, logger=None)
    # Calculate the coverage: the proportion of actual values within the predicted intervals
    coverage = np.mean([low <= actual <= high for actual, (low, high) in zip(y_test, intervals)])
    assert (
        coverage >= interval_coverage_limit
    ), f"Sales price is not within prediction interval for at least {interval_coverage_limit * 100}% of cases, coverage {coverage}"


def test_time_predict(testdata, time_limit_ms=250, quantiles=[0.05, 0.5, 0.95], subset_fraction=0.2):
    """Test that predicting for a random subset of X_test takes less than a certain time limit in milliseconds."""
    model_store, X_train, X_test, y_train, y_test = testdata

    subset_size = int(len(X_test) * subset_fraction)
    random_indices = random.sample(range(len(X_test)), subset_size)

    X_test_subset = X_test.iloc[random_indices]

    start_time = time.time()
    strategy, preds, intervals, error = predict_interval(X_test_subset, model_store.regressor, quantiles, logger=None)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

    assert elapsed_time < time_limit_ms, f"Prediction for a subset took longer than {time_limit_ms}ms"


def test_predict(testdata, quantiles=[0.05, 0.5, 0.95], subset_fraction=0.2):
    """Test that predicting for a random subset of X_test yields a result."""
    model_store, X_train, X_test, y_train, y_test = testdata

    subset_size = int(len(X_test) * subset_fraction)
    random_indices = random.sample(range(len(X_test)), subset_size)

    X_test_subset = X_test.iloc[random_indices]

    strategy, preds, intervals, error = predict_interval(X_test_subset, model_store.regressor, quantiles, logger=None)

    assert len(preds) == len(X_test_subset), "Prediction did not yield a result for each sample in the subset"
