"""App for price"""

# get env variable
import os

# fastapi
from fastapi import FastAPI, Request, Response, HTTPException, status, Header
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Union, List

# periodical data read-in
from threading import Thread

import pandas as pd
import numpy as np
import time

# config file
import configparser

# get loggers
from buycycle.logger import Logger

# sql queries, feature selection and other functions from src
from src.data import ModelStore, feature_engineering
from src.strategies import GenericStrategy
from src.helper import construct_input_df
from src.driver import msrp_min, msrp_max

config_paths = "config/config.ini"
config = configparser.ConfigParser()
config.read(config_paths)

path = "data/"

# app = Flask(__name__)
app = FastAPI()
# read the environment from the docker environment variable
environment = os.getenv("ENVIRONMENT")
ab = os.getenv("AB")
app_name = "price"
app_version = "stable-002-highprice"

logger = Logger.configure_logger(environment, ab, app_name, app_version)

logger.info("FastAPI app started")

# create data stores and load periodically
model_store = ModelStore()

# inital data readin
while True:
    try:
        model_store.read_data()
        break
    except Exception as e:
        logger.error("Data could not initially be read, trying in 60sec")
        time.sleep(60)

# then read the data periodically
model_loader = Thread(target=model_store.read_data_periodically, args=(720, logger))

model_loader.start()


class PriceRequest(BaseModel):
    """Class representing the price request, the order need to be identical with the order in driver.py"""
    template_id: Union[int, None] = None
    brake_type_code: Union[object, None] = None
    frame_material_code: Union[object, None] = None
    shifting_code: Union[object, None] = None
    color: Union[object, None] = None
    bike_category_id: Union[int, None] = None
    motor: Union[int, None] = None
    sales_country_id: Union[int, None] = None
    msrp: Union[float, None] = None
    condition_code: Union[object, None] = None
    bike_created_at_year: Union[int, None] = None
    bike_created_at_month: Union[int, None] = None
    rider_height_min: Union[float, None] = None
    rider_height_max: Union[float, None] = None
    sales_duration: Union[int, None] = None
    bike_year: Union[int, None] = None
    is_mobile: Union[int, None] = None
    is_ebike: Union[int, None] = None
    is_frameset: Union[int, None] = None
    mileage_code: Union[object, None] = None
    bike_type_id: Union[int, None] = None
    bike_component_id: Union[int, None] = None
    family_model_id: Union[int, None] = None
    family_id: Union[int, None] = None
    brand_id: Union[int, None] = None  

@app.get("/")
async def home():
    return {"msg": "price"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/price_interval")
async def price_interval(
    request_data: Union[PriceRequest, List[PriceRequest]],
    strategy: str = Header(default="Generic"),
):
    """
    take in bike data
    the payload should be in PriceRequest format
    """
    # Convert the PriceRequest to a dataframe
    if isinstance(request_data, list):
        request_dic = []
        for r in request_data:
            request_dic.append(r.model_dump(exclude_unset=True))
        price_payload = pd.DataFrame(request_dic)
    else:
        request_dic = request_data.model_dump(exclude_unset=True)
        price_payload = pd.DataFrame([request_dic])

    if price_payload.empty:
        return Response(
            content="no valid request values", status_code=status.HTTP_400_BAD_REQUEST
        )

    # get target strategy, with default value "generic"
    strategy_target = strategy

    # fill the missing data with np.nan and do feature engineering
    features = list(PriceRequest.model_fields.keys())
    X_constructed = construct_input_df(price_payload, features)
    X_feature_engineered = feature_engineering(X_constructed)

    # Predict the price and interval
    with model_store._lock:
        # Split the dataframe, df1 with msrp < msrp_min, df2 with msrp_min <= msrp <= msrp_max or nan, df3 with msrp > msrp_max
        df1 = X_feature_engineered[X_feature_engineered["msrp"] < msrp_min].copy()
        df2 = X_feature_engineered[
            (
                (X_feature_engineered["msrp"] >= msrp_min)
                & (X_feature_engineered["msrp"] <= msrp_max)
            )
            | pd.isna(X_feature_engineered["msrp"])
        ].copy()
        df3 = X_feature_engineered[X_feature_engineered["msrp"] > msrp_max].copy()

        # Step 2: Store the original indices, later for the combination
        df1_original_indices = df1.index
        df2_original_indices = df2.index
        df3_original_indices = df3.index

        df1_price = []
        df1_interval = []
        df2_price = []
        df2_interval = []
        df3_price = []
        df3_interval = []

        # For the low price bike, set preds = msrp * ratio
        if not df1.empty:
            df1_price = [round(p, 2) for p in (df1["msrp"] * 0.68).tolist()]
            df1_interval = [
                [round(p - p * 0.25, 2), round(p + p * 0.25, 2)] for p in df1_price
            ]

        # For other bikes, use ML model
        if not df2.empty:
            generic_strategy = GenericStrategy(
                model_store.regressor, model_store.data_transform_pipeline, logger
            )
            quantiles = [0.2, 0.5, 0.8]

            X_transformed = model_store.data_transform_pipeline.transform(df2)
        
            strategy, df2_price, df2_interval, error = generic_strategy.predict_price(
                X=X_transformed, quantiles=quantiles
            )

        # For the high price bike, set preds = msrp * ratio
        if not df3.empty:
            df3_price = [round(p, 2) for p in (df3["msrp"] * 0.6).tolist()]
            df3_interval = [
                [round(p - p * 0.25, 2), round(p + p * 0.25, 2)] for p in df3_price
            ]

        # Concatenate the price and interval lists back together
        price = df1_price + df2_price + df3_price
        interval = df1_interval + df2_interval + df3_interval

        # Sort by the original indices
        original_indices = np.concatenate(
            [df1_original_indices, df2_original_indices, df3_original_indices]
        )
        price = [x for _, x in sorted(zip(original_indices, price))]
        interval = [x for _, x in sorted(zip(original_indices, interval))]

        logger.info(
            strategy,
            extra={
                "price": price,
                "interval": interval,
                "quantiles": quantiles,
                "X_input": request_dic,
            },
        )

        if error:
            # Return error response if it exists
            logger.error(
                "Error no price prediction available, exception: {}".format(error)
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Price prediction not available",
            )

        else:

            # Return success response with recommendation data and 200 OK
            return {
                "status": "success",
                "strategy_target": strategy_target,
                "strategy": strategy,
                "quantiles": quantiles,
                "price": price,
                "interval": interval,
                "app_name": app_name,
                "app_version": app_version,
            }


# Error handling for 400 Bad Request
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the error details using the provided logger
    logger.error(
        "400 Bad Request: {}".format(str(exc)),
        extra={
            "info": "Invalid request body format",
        },
    )
    # Construct a hint for the expected request body format
    expected_format = PriceRequest.model_json_schema
    # Return a JSON response with the error details and the hint
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "message": f"400 Bad Request: {str(exc)}",
            "hint": "The provided request body format is incorrect. "
            "Please ensure it adheres to the expected format:",
            "expected_format": expected_format,
        },
    )


# add 500 error handling
@app.exception_handler(500)
def internal_server_error_handler(request: Request, exc: HTTPException):
    """Log the error details using the provided logger"""
    logger.error(
        f"500 Internal Server Error: {str(exc)}",
        extra={
            "info": "Internal server error",
        },
    )
    # Return a JSON response with the error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "Internal Server Error: " + str(exc)},
    )
