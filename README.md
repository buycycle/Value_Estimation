# price

Model that predicts the sales price of a bicycle given a set of bicycle features and historic sales data.
The model also returns a prediction interval.

## Model
Currently, the best model performance is shown by tree based regression  models.
A ExtraTreesRegressor showed sub 10% MAPE when tested on current months sales.

## Imputation
The model needs to be able to cope with different data quality for different use cases. We employ an MissForest Imputer that is able to deal with numeric and categorical missing features.

## Prediction interval
The mode returns the prediction interval for a given confidence score.
In this context, a prediction interval is a range of values that is likely to contain the true value of a target variable for a given test instance, with a specified level of confidence.
see: Nicolai Meinshausen. Quantile Regression Forests. Journal of Machine Learning Research, 7(6), 983-999, 2006. URL: https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf.

# AB test

The AB test is implemented with istio on Kubernetes. The request are assigned randomly with a certain weight.
There are three scenarios:

If traffic should be routed to the dev environment, name them -dev for the version name.
This allows the Load Balancer to route to the dev environment.

    Test two versions:

    Prepare two versions and add -stable, -canary to the app_version

    Define your image tag and name and weight under Values.api.version.
    use the same name as in the model app_version
    Sync Kubernetes.

    After test finished:

    set canary to 0% to stop allocation of new people

    and/or rename Values.api.version.name to force also already assigned to 0%.

    Change model image but keep user assignment:

    This only works if the model input data is not changing.

    Set Values.newVersion to false and change Values.api.version.tag to different value.
    Sync Kubernets.

    Start a new ab test:

    same as 'test two versions'
    The PVCs and cronjobs for the data creation are kept and assigned through meta_name, so change the metaname.
    If data requirements are chaning for the model version first start version with
    0 value and wait until the old data from the pvc is delete and new one created.
    Then set to desired weight.



## Requirements

* [Docker version 20 or later](https://docs.docker.com/install/#support)

## Setup development environment

We setup the conda development environment with the following command.

- `make setup`

Install requirements

- `make install`

## Lint and formatting

- `make lint`

- `make format`


## Docker

when creating an docker image the data is downloaded and prepared. Build test and production stages and runs tests.

- `docker compose build`

Run app.

- `docker compose up app`


## Driver and Config

src/driver.py defines the SQL queries, categorical and numerical features as well as the prefilter_features.
config/config.ini holdes DB credentials.

## Endpoint

REST API

### Price Interval

	Path: /price_interval
	HTTP Verb: POST

Description: Return a an np.array of price and interval for a given input X_input. Either for one bike or multiple bikes.

Parameters:
    A JSON object with the following keys and data types. Each dictionary represents a bike.
    If a non-complete set of parameters is supplied the model imputs the rest of the data:

    ```json
    [{
        "template_id": "Int64",
        "msrp": "Float64",
        "bike_created_at_year": "Int64",
        "bike_created_at_month": "Int64",
        "bike_year": "Int64",
        "sales_duration": "Int64",
        "sales_country_id": "Int64",
        "bike_type_id": "Int64",
        "bike_category_id": "Int64",
        "mileage_code": "object",
        "motor": "Int64",
        "city": "object",
        "condition_code": "object",
        "frame_size": "object",
        "rider_height_min": "Float64",
        "rider_height_max": "Float64",
        "brake_type_code": "object",
        "frame_material_code": "object",
        "shifting_code": "object",
        "bike_component_id": "Int64",
        "color": "object",
        "family_model_id": "Int64",
        "family_id": "Int64",
        "brand_id": "Int64",
        "quality_score": "Int64",
        "is_mobile": "Int64",
        "currency_id": "Int64",
        "seller_id": "Int64",
        "is_ebike": "Int64",
        "is_frameset": "Int64"
    },
    {
        "template_id": "Int64",
        ...
    }
    ]

    ```
Headers:
    strategy: which strategy to use

Example:
    ```bash
    
     curl -i -X POST price.buycycle.com/price_interval \
    -H "Content-Type: application/json" \
    -H "strategy: Generic" \
    -d '[{"family_id": 12, "msrp": 1200}, {"family_id": 2, "msrp": 2200}]'
    
    ```

Return:
    A JSON response with the following structure:
    ```json
    {
        "status": "success",
        "strategy": "Generic",
        "quantiles": [0.05, 0.5, 0.95],
        "price": [2382.0, ...],
        "interval": [[1600.0, 3000.0],[...]],
        "app_name": "price",
        "app_version": "stable-001"
    }
    ```
    
HTTP Status Codes:

    - `200`: Successful response with price prediction data.
    - `400`: Bad request due to invalid input parameters.
    - `404`: Price prediction failed due to an error in the prediction process.
Note: The `strategy` field in the response is a string indicating the prediction strategy used. The `quantiles` field represents the quantiles for which the price intervals are calculated. The `price` field is a list containing the predicted price, and the `interval` field is a list of lists, with each sublist representing a predicted price interval.
