"""App for price"""
# get env variable
import os

# flask
from flask import Flask, request, jsonify

# periodical data read-in
from threading import Thread

import pandas as pd

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

app = Flask(__name__)
# read the environment from the docker environment variable
environment = os.getenv("ENVIRONMENT")
ab = os.getenv("AB")
app_name = "price"
app_version = 'stable-001'

KAFKA_TOPIC = config["KAFKA"]["topic_price"]
KAFKA_BROKER = config["KAFKA"]["broker"]

logger = Logger.configure_logger(environment, ab, app_name, app_version)
logger = KafkaLogger(environment, ab, app_name,
                     app_version, KAFKA_TOPIC, KAFKA_BROKER)

logger.info("Flask app started")

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


@app.route("/")
def home():
    html = f"<h3>price</h3>"
    return html.format(format)


@app.route("/price_interval", methods=["POST"])
def price():
    """take in bike data
    the payload should be in the following format:

    {
        'template_id': 'Int64',
        'msrp': 'Float64',
        'bike_created_at_year': 'Int64',
        'bike_created_at_month': 'Int64',
        'bike_year': 'Int64',
        'sales_duration': 'Int64',
        'sales_country_id': 'Int64',
        'bike_type_id': 'Int64',
        'bike_category_id': 'Int64',
        'mileage_code': 'object',
        'motor': 'Int64',
        'city': 'object',
        'condition_code': 'object',
        'frame_size': 'object',
        'rider_height_min': 'Float64',
        'rider_height_max': 'Float64',
        'brake_type_code': 'object',
        'frame_material_code': 'object',
        'shifting_code': 'object',
        'bike_component_id': 'Int64',
        'color': 'object',
        'family_model_id': 'Int64',
        'family_id': 'Int64',
        'brand_id': 'Int64',
        'quality_score': 'Int64',
        'is_mobile': 'Int64',
        'currency_id': 'Int64',
        'seller_id': 'Int64',
        'is_ebike': 'Int64',
        'is_frameset': 'Int64'
    }
    """

    # Get the JSON payload from the request
    json_payload = request.json
    # Check if the payload is a list of dictionaries (multiple bikes)
    if isinstance(json_payload, list):
        # Convert the list of dictionaries to a pandas DataFrame
        price_payload = pd.DataFrame(json_payload)
    else:
        # If it's a single dictionary (one bike), convert it to a DataFrame with an index
        price_payload = pd.DataFrame([json_payload])

    # get target strategy, currently not implemented since we only have generic strategy
    strategy_target = request.headers.get('strategy', 'NA')  # Provide a default value if not found




    features = ['template_id', 'msrp', 'bike_created_at_year', 'bike_created_at_month',
                'bike_year', 'sales_duration', 'sales_country_id', 'bike_type_id',
                'bike_category_id', 'mileage_code', 'motor', 'city', 'condition_code',
                'frame_size', 'rider_height_min', 'rider_height_max', 'brake_type_code',
                'frame_material_code', 'shifting_code', 'bike_component_id', 'color',
                'family_model_id', 'family_id', 'brand_id', 'quality_score',
                'is_mobile', 'currency_id', 'seller_id', 'is_ebike', 'is_frameset']

    #filter out non features, in the payload
    X_input = price_payload[price_payload.columns.intersection(features)]

    X_constructed = construct_input_df(X_input, features)

    with model_store._lock:
        generic_strategy = GenericStrategy(
            model_store.regressor, model_store.data_transform_pipeline, logger)

        quantiles = [0.2, 0.5, 0.8]

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
        return (
            jsonify(
                {"status": "error", "message": "Price prediction not available"}),
            404,
        )

    else:
        # Return success response with recommendation data and 200 OK
        return (
            jsonify(
                {
                    "status": "success",
                    "strategy_target": strategy_target,
                    "strategy": strategy,
                    "quantiles": quantiles,
                    "price": price,
                    "interval": interval,
                    "app_name": app_name,
                    "app_version": app_version,
                }
            ),
            200,
        )


# test this out, which erros do we need to handle


# Error handling for 400 Bad Request
@app.errorhandler(400)
def bad_request_error(e):
    # Log the error details using the provided logger
    logger.error(
        "400 Bad Request:",
        extra={
            "info": "user_id, bike_id and n must be convertable to integers",
        },
    )

    return (
        jsonify({"status": "error",
                "message": "Bad Request, user_id, bike_id and n must be convertable to integers"}),
        400,
    )


# add 500 error handling

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
