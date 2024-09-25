"""
functional tests for the fastapi model response
"""

import time
from fastapi.testclient import TestClient
from tests.test_fixtures import app_mock, logger_mock
import numpy as np


def test_single_request_fastapi(app_mock, limit=150):
    """test the single price request for all strategies of the fastapi app"""
    client = TestClient(app_mock)
    request = {"msrp": 1200, "family_id": 12, "is_ebike": 1}

    start_time = time.time()
    response = client.post("/price_interval", json=request)
    end_time = time.time()

    # ensure the request was successful
    assert (
        response.status_code == 200
    ), f"request failed with status code {response.status_code} for request {request}"
    # parse the response data
    data = response.json()

    strategy_used = data.get("strategy")
    price = data.get("price")
    interval = data.get("interval")

    # check the time taken for the recommendation
    assert (
        end_time - start_time < limit
    ), f"{strategy_used} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
    # assert that the response has the expected length
    assert (
        len(price) == 1
    ), f"expected 1 price for strategy {strategy_used}, got the price {price}"
    assert (
        len(interval) == 1
    ), f"expected 2 interval for strategy {strategy_used}, got the interval {interval}"


def test_multiple_request_fastapi(app_mock, limit=150):
    """test time and len of return for all strategies of the fastapi app"""
    client = TestClient(app_mock)
    request = [
        {"template_id": 9973, "msrp": 2499, "frame_material_code": "carbon"},
        {"template_id": 14045, "msrp": 5499, "family_model_id": 8258},
        {"template_id": 14045, "msrp": 0, "family_model_id": 8258},
        {"template_id": 14045, "msrp": None, "family_model_id": 8258},
        {"template_id": 14045, "family_model_id": 8258},
        {"brake_type_code":"hydraulic","frame_material_code":"carbon","shifting_code":"electronic","bike_category_id":1,"bike_created_at_month":9,"msrp":4299.0,"condition_code":3,"bike_created_at_year":2024,"sales_duration":14,"is_mobile":1,"is_ebike":0,"is_frameset":0,"bike_type_id":1,"bike_component_id":64,"family_model_id":117380,"family_id":11718,"brand_id":128,"bike_year":2023}, 
        {"brake_type_code":"hydraulic","frame_material_code":"carbon","shifting_code":"mechanical","bike_category_id":1,"bike_created_at_month":9,"msrp":3299.0,"condition_code":3,"bike_created_at_year":2024,"sales_duration":14,"is_mobile":0,"is_ebike":0,"is_frameset":0,"bike_type_id":1,"bike_component_id":17,"family_model_id":8758,"family_id":2371,"brand_id":227,"bike_year":2021}
    ]

    start_time = time.time()
    response = client.post("/price_interval", json=request)
    end_time = time.time()

    # ensure the request was successful
    assert (
        response.status_code == 200
    ), f"request failed with status code {response.status_code} for request {request}"
    # parse the response data
    data = response.json()

    strategy_used = data.get("strategy")
    price = data.get("price")
    interval = data.get("interval")

    # check the time taken for the recommendation
    assert (
        end_time - start_time < limit
    ), f"{strategy_used} took {(end_time - start_time)*1000} ms, limit is {limit*1000} ms"
    # assert that the response has the expected length
    assert (
        len(price) == 7
    ), f"expected 7 price for strategy {strategy_used}, got the price {price}"
    assert (
        len(interval) == 7
    ), f"expected 7 interval for strategy {strategy_used}, got the interval {interval}"
    
    msrp_list = [r.get("msrp") for r in request]   
    for p, msrp in zip(price, msrp_list):
        if msrp is not None and not np.isnan(float(msrp)) and msrp != 0:
            assert (
                p < msrp*0.9 and p > msrp*0.3
            ), f"expected price < msrp, got {p} >= {msrp}"

