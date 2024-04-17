"""
functional tests for the fastapi model response
"""

import time
from fastapi.testclient import TestClient
from tests.test_fixtures import app_mock, logger_mock


def test_single_request_fastapi(app_mock, limit=150):
    """test the single price request for all strategies of the fastapi app"""
    client = TestClient(app_mock)
    request = {"family_id": 12, "msrp": 1200, "is_ebike": 1}

    start_time = time.time()
    response = client.post("/price_interval", json=request)
    end_time = time.time()

    # ensure the request was successful
    assert response.status_code == 200, f"request failed with status code {response.status_code} for request {request}"
    # parse the response data
    data = response.json()

    strategy_used = data.get("strategy")
    price = data.get("price")
    interval = data.get("interval")

    # check the time taken for the recommendation
    assert end_time - start_time < limit, f"{strategy_used} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
    # assert that the response has the expected length
    assert len(price) == 1, f"expected 1 price for strategy {strategy_used}, got the price {price}"
    assert len(interval) == 1, f"expected 2 interval for strategy {strategy_used}, got the interval {interval}"


def test_multiple_request_fastapi(app_mock, limit=150):
    """test time and len of return for all strategies of the fastapi app"""
    client = TestClient(app_mock)
    request = [{"template_id": 9973, "msrp": 3299, "frame_material_code": "carbon"}, {"family_id": 821, "msrp": 3500, "motor": 1}]

    start_time = time.time()
    response = client.post("/price_interval", json=request)
    end_time = time.time()

    # ensure the request was successful
    assert response.status_code == 200, f"request failed with status code {response.status_code} for request {request}"
    # parse the response data
    data = response.json()

    strategy_used = data.get("strategy")
    price = data.get("price")
    interval = data.get("interval")

    # check the time taken for the recommendation
    assert ( 
        end_time - start_time < limit, 
        f"{strategy_used} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
    )
    # assert that the response has the expected length
    assert (
        len(price) == 2, 
        f"expected 2 price for strategy {strategy_used}, got the price {price}"
    )
    assert (
        len(interval) == 2, 
        f"expected 2 interval for strategy {strategy_used}, got the interval {interval}"
    )    
