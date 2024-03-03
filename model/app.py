"""App for price"""
# get env variable
import os

# fastapi
from fastapi import FastAPI, Request, HTTPException, status, Body, Header
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator

# periodical data read-in
from threading import Thread

import pandas as pd
import time

# config file
import configparser

# get loggers
from buycycle.logger import Logger
from buycycle.logger import KafkaLogger


# sql queries and feature selection
from src.driver import *

from src.data import ModelStore

# import the function from src
from src.strategies import GenericStrategy

from src.helper import construct_input_df, get_field_value

config_paths = "config/config.ini"

config = configparser.ConfigParser()
config.read(config_paths)

path = "data/"

app = FastAPI()
# read the environment from the docker environment variable
environment = os.getenv("ENVIRONMENT")
ab = os.getenv("AB")
app_name = "price"
app_version = 'stable-001'

KAFKA_TOPIC = config["KAFKA"]["topic"]
KAFKA_BROKER = config["KAFKA"]["broker"]

logger = Logger.configure_logger(environment, ab, app_name, app_version)
#logger = KafkaLogger(environment, ab, app_name,
#                     app_version, KAFKA_TOPIC, KAFKA_BROKER)

logger.info("FastAPI app started")

# create data stores and load periodically
model_store = ModelStore()

# inital data readin
while True:
    try:
        model_store.read_data()
        break
    except Exception as e:
        logger.error("Data could not initially be red, trying in 60sec")
        time.sleep(60)

# then read the data periodically
model_loader = Thread(
    target=model_store.read_data_periodically, args=(720, logger))

model_loader.start()

class PriceRequest(BaseModel):
    template_id: int | None = None
    msrp: float | None = None
    bike_created_at_year: int | None = None
    bike_created_at_month: int | None = None
    bike_year: int | None = None
    sales_duration: int | None = None
    sales_country_id: int | None = None
    bike_type_id: int | None = None
    bike_category_id: int | None = None
    mileage_code: str | None = None
    motor: int | None = None
    city: str | None = None
    condition_code: str | None = None
    frame_size: str | None = None
    rider_height_min: float | None = None
    rider_height_max: float | None = None
    brake_type_code: int | None = None
    frame_material_code: str | None = None
    shifting_code: str | None = None
    bike_component_id: int | None = None
    color: str | None = None
    family_model_id: int | None = None
    family_id: int | None = None
    brand_id: int | None = None
    quality_score: int | None = None
    is_mobile: int | None = None
    currency_id: int | None = None
    seller_id: int | None = None
    is_ebike: int | None = None
    is_frameset: int | None = None

    @validator("*", pre=True, always=True)
    def at_least_one_value(cls, values):
        if len(values) < 1:
            raise ValueError("At least one attribute must be provided in the request body.")
        return values

@app.get("/")
def home():
    html = "<h3>price</h3>"
    return html


@app.post("/price_interval")
def price(request_data: list[PriceRequest] = Body(), strategy: str= Header(default='NA')):
    """take in bike data
    the payload should be in PriceRequest format
    """
    # Convert the list of PriceRequest to a dataframe
    price_payload = pd.DataFrame(request_data)

    # get target strategy, currently not implemented since we only have generic strategy
    strategy_target = strategy  # Provide a default value if not found

    features = list(PriceRequest.model_fields.keys())

    #filter out non features, in the payload
    X_input = price_payload[price_payload.columns.intersection(features)]

    X_constructed = construct_input_df(X_input, features)

    with model_store._lock:
        generic_strategy = GenericStrategy(
            model_store.regressor, model_store.data_transform_pipeline, logger)

        quantiles = [0.05, 0.5, 0.95]

        X_transformed = model_store.data_transform_pipeline.transform(X_constructed)

        strategy, price, interval, error = generic_strategy.predict_price(
            X=X_transformed, quantiles=quantiles)

        price = price.tolist()
        interval= interval.tolist()

    logger.info(
        strategy,
        extra={
            "X_input": X_input.to_dict(orient='records'),
            "price": price,
            "interval": interval,
            "quantiles": quantiles,
        },
    )
    if error:
        # Return error response if it exists
        logger.error(
            "Error no price prediction available, exception: " + error)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Price prediction not available")

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

# test this out, which erros do we need to handle


# Error handling for 400 Bad Request
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the error details using the provided logger
    logger.error(
        "400 Bad Request:",
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
            "message": "Invalid request body format",
            "hint": "The provided request body format is incorrect. Please ensure it adheres to the expected format:",
            "expected_format": expected_format,
        },
    )


# add 500 error handling
@app.exception_handler(500)
def internal_server_error_handler(request: Request, exc: HTTPException):
    # Log the error details using the provided logger
    logger.error(
        "500 Internal Server Error: " + str(exc),
        extra={
            "info": "Internal server error",
        },
    )
    # Return a JSON response with the error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "Internal Server Error: " + str(exc)},
    )

