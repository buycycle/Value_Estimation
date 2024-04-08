"""
functional tests for the fastapi model response
"""

import time

from fastapi.testclient import TestClient
from model.app import app
from unittest.mock import Mock, patch
from buycycle.logger import Logger
import pytest

@pytest.fixture(scope="package")
def logger_mock():
    "mock the Logger"
    # Create a mock Logger instance
    logger_mock = Mock(spec=Logger)

    return logger_mock

@pytest.fixture(scope="package")
def app_mock(logger_mock):
    "patch the model with the logger mock version and prevent threads from starting"

    with patch("buycycle.logger.Logger", return_value=logger_mock), patch(
        "src.data.ModelStore.read_data"):
        # The above patches will replace the actual methods with mocks that do nothing
        from model.app import app  # Import inside the patch context to apply the mock

        yield app  # Use yield to make it a fixture

def test_read_main(app_mock):
    client = TestClient(app_mock)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "test"}, f"expected response {response}"


def test_integration_fastapi(app_mock):
    """test time and len of return for all strategies of the fastapi app"""
    client = TestClient(app_mock)
    request = {"family_id": 12, "msrp": 1200}

    response = client.post("/price_interval", json=request)

    # ensure the request was successful
    assert response.status_code == 200, f"request failed with status code {response.status_code} for request {request}"
    # parse the response data
    data = response.json()

    strategy_used = data.get("strategy")
    price = data.get("price")
    interval = data.get("interval")



    # # check the time taken for the recommendation
    # assert end_time - start_time < limit, f"{strategy_used} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
    # # assert that the response has the expected length n
    assert len(price) == 1, f"expected price and interval for strategy {strategy_used}, got {price} and {interval}"
    # # ... (other assertions or checks based on the response)