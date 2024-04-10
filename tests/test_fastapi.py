"""
functional tests for the fastapi model response
"""

import time
from fastapi.testclient import TestClient

def test_read_main(app_mock):
    client = TestClient(app_mock)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "price"}, f"expected response {response}"


def test_integration_fastapi(app_mock, limit=150):
    """test time and len of return for all strategies of the fastapi app"""
    client = TestClient(app_mock)
    request = [{"family_id": 12, "msrp": 1200}, {"family_id": 2, "msrp": 2200}]

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
    # assert that the response has the expected length n
    assert len(price) == 2, f"expected 2 price for strategy {strategy_used}, got the price {price}"
    assert len(interval) == 2, f"expected 2 interval for strategy {strategy_used}, got the interval {interval}"

def test_exception_handler_request(app_mock):
    client = TestClient(app_mock)
    request = {"family_id": 2, "msrp": 2200}
    response = client.post("/price_interval", json=request)
    assert response.status_code == 400, f"request failed with status code {response.status_code} for request {request}"
