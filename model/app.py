"""App for price"""

# get env variable
import os

# fastapi
from fastapi import FastAPI, Request, Response, HTTPException, status, Header
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator
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
from src.price import predict_with_msrp

config_paths = "config/config.ini"
config = configparser.ConfigParser()
config.read(config_paths)

path = "data/"

app = FastAPI()
# read the environment from the docker environment variable
environment = os.getenv("ENVIRONMENT")
ab = os.getenv("AB")
app_name = "price"
app_version = "canary-003-interval_10"

logger = Logger.configure_logger(environment, ab, app_name, app_version)
logger.info("FastAPI app started")

# create data stores and load periodically
model_store = ModelStore()

# inital data readin
while True:
    try:
        model_store.read_data() # load model and data_transform_pipeline
        break
    except Exception as e:
        logger.error("Data could not initially be read, trying in 60sec")
        time.sleep(60)

# then read the data periodically in 2880 minutes(2 days), try block included in read_data_periodically in DataStoreBase class
model_loader = Thread(target=model_store.read_data_periodically, args=(2880, logger))
model_loader.start()
 

class PriceRequest(BaseModel):
    """Class representing the price request, the order need to be identical with the order in driver.py"""
    # template_id: Union[int, None] = None
    brake_type_code: Union[object, None] = None
    frame_material_code: Union[object, None] = None
    shifting_code: Union[object, None] = None
    color: Union[object, None] = None
    bike_category_id: Union[int, None] = None
    motor: Union[int, None] = None
    sales_country_id: Union[int, None] = None
    bike_created_at_month: Union[int, None] = None
    msrp: Union[float, None] = None
    condition_code: Union[object, None] = None
    bike_created_at_year: Union[int, None] = None
    rider_height_min: Union[float, None] = None
    rider_height_max: Union[float, None] = None
    sales_duration: Union[int, None] = None
    is_mobile: Union[int, None] = None
    is_ebike: Union[int, None] = None
    is_frameset: Union[int, None] = None
    mileage_code: Union[object, None] = None
    bike_type_id: Union[int, None] = None
    bike_component_id: Union[int, None] = None
    family_model_id: Union[int, None] = None
    family_id: Union[int, None] = None
    brand_id: Union[int, None] = None
    bike_year: Union[int, None] = None

    @validator("msrp", pre=True)
    def validate_msrp(cls, value):
        if value is None:
            return 0
        if isinstance(value, str):
            try:
                # Attempt to convert the string to an integer
                return int(value)
            except ValueError:
                # If conversion fails, return the default value
                return 0
        return value


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
    # get target strategy, with default value "generic"
    strategy_target = strategy

    # Convert the PriceRequest to a dataframe
    try:
        if isinstance(request_data, list):
            request_dic = [r.model_dump(exclude_unset=True) for r in request_data]
            price_payload = pd.DataFrame(request_dic)
        else:
            request_dic = request_data.model_dump(exclude_unset=True)
            price_payload = pd.DataFrame([request_dic])

        if price_payload.empty:
            logger.error("Request received: The payload is empty.")  
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid request values")
    except Exception as e:
        logger.error("Error processing request data: %s wich x_input: %s", str(e), request_dic)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request data format")

    # construct dataframe, fill the missing data with np.nan and do feature engineering
    try:
        features = list(PriceRequest.model_fields.keys())
        X_constructed = construct_input_df(price_payload, features)
        X_feature_engineered = feature_engineering(X_constructed)
    except Exception as e:
        logger.error("Error in feature engineering: %s wich x_input: %s", str(e), request_dic)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Feature engineering failed")

    # Predict the price and interval
    try:
        with model_store._lock:
            # Split the data into parts according to msrp
            # based on the original msrp before inflation adjustment(feature engineering)
            mask_msrp = (
                (X_constructed["msrp"] > 0) & (X_constructed["msrp"] <= (msrp_min))
            ) | (X_constructed["msrp"] >= msrp_max)
            mask_model = ~mask_msrp | pd.isna(X_constructed["msrp"])
            conditions = [mask_msrp, mask_model]

            # Define choice function for the ML model predition cases
            generic_strategy = GenericStrategy(
                model_store.regressor, model_store.data_transform_pipeline, logger
            )
            quantiles = [0.2, 0.5, 0.8]

            X_transformed = model_store.data_transform_pipeline.transform(
                X_feature_engineered
            )

            strategy, price, interval, error = generic_strategy.predict_price(
                X=X_transformed, quantiles=quantiles
            )
            combined = list(zip(price, interval))
            predictions = pd.Series(combined)

            # Define choices, which need to return the same structure
            choices = [
                X_constructed.apply(predict_with_msrp, args=(0.5,), axis=1),
                predictions,
            ]

            # Define default value
            default = pd.Series([(np.nan, [np.nan, np.nan])] * len(X_feature_engineered))

            # apply the conditions and get the price and interval
            X_feature_engineered["price"], X_feature_engineered["interval"] = zip(
                *np.select(conditions, choices, default=default)
            )

            # sort by original request ids
            X_feature_engineered = X_feature_engineered.reindex(X_constructed.index)
            # Extract the price and interval columns
            price = X_feature_engineered["price"].tolist()
            interval = X_feature_engineered["interval"].tolist()

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
                logger.error("Prediction error: %s", error)
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
    except Exception as e:
        logger.error("Error during prediction process: %s wich x_input: %s", str(e), request_dic)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed")


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
async def internal_server_error_handler(request: Request, exc: HTTPException):
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

    
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Log the HTTPException details
    logger.error(f"HTTP Exception: {str(exc.detail)}", extra={
        "info": "HTTP Exception caught",
    })

    # Return the HTTPException's status code and detail in the response
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error details
    logger.error(f"Unexpected error: {str(exc)}", extra={
        "info": "Unexpected error caught",
    })

    # Return a generic 500 Internal Server Error response
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "An unexpected error occurred. Please try again later."}
    )


