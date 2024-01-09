import os
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import category_encoders as ce
from joblib import dump, load
from buycycle.data import sql_db_read, DataStoreBase
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor



def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model: Callable,
    target: str,
    parameters: Optional[Dict[str, Union[int, float]]] = None,
    scoring: Callable = mean_absolute_percentage_error,
) -> Tuple[Callable, Callable]:
    """
    Trains the model.
    Args:
        X_train: Dataframe of transformed training data.
        y_train: Dataframe of target training data.
        model: Model to train.
        target: Target variable.
        parameters: Parameters for the model. Default is None.
        scoring: Scoring function. Default is mean_absolute_percentage_error.
    Returns:
        regressor: Trained model.
    """
    if parameters is None:
        parameters = {"max_depth": 50, "n_estimators": 200}

    regressor = model()
    if parameters:
        regressor.set_params(**parameters)
    regressor.fit(X_train, y_train)

    # Check if the model is a quantile regressor
    if isinstance(regressor, (RandomForestQuantileRegressor, ExtraTreesQuantileRegressor)):
        preds = regressor.predict(X_train, quantiles=[0.5])
    else:
        preds = regressor.predict(X_train)

    score = scoring(y_train, preds)

    print("{} Train error: {}".format(model, score))

    return regressor


def test(
    X_transformed: pd.DataFrame,
    y: pd.Series,
    regressor: Callable,
    data_transform_pipeline: Callable,
    scoring: Callable = mean_absolute_percentage_error,
    quantiles: List[float] = [0.025, 0.5, 0.975],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Tests the model.
    Args:
        X_transformed: Transformed Test features.
        y: Test target.
        regressor: Trained model.
        data_transform_pipeline: Data processing pipeline.
        scoring: Scoring function. Default is mean_absolute_percentage_error.
        quantiles: Quantiles for prediction intervals. Default is [0.025, 0.5, 0.975].
    Returns:
        preds: Predictions.
        interval: Prediction intervals.
        error
    """
    # Check if the model is a quantile regressor
    if isinstance(regressor, (RandomForestQuantileRegressor, ExtraTreesQuantileRegressor)):
        strategy, preds, interval, error = predict_interval(X_transformed, regressor, quantiles, logger=None)

    else:
        preds = predict_point_estimate(X_transformed, regressor)
        interval = None
    pd.DataFrame(preds).to_pickle("data/preds.pkl")

    score = scoring(y, preds)
    print("Error: {}".format(score))

    return strategy, preds, interval, error


def predict_point_estimate(X_transformed: pd.DataFrame, regressor: Callable) -> np.ndarray:
    """
    Transform X and predicts target variable.
    Args:
        X_transformed: Transformed Features.
        regressor: Trained model.
    Returns:
        preds: Predictions.
    """
    preds = regressor.predict(X_transformed)

    return preds


def predict_interval(
        X_transformed: pd.DataFrame, regressor: Callable, quantiles: List[float], logger: Callable
) -> Tuple[str, np.ndarray, np.ndarray, str]:
    """
    Transform X and predicts target variable as well as prediction interval.
    Args:
        X_transformed: Transformed Features.
        regressor: Trained model.
        quantiles: Quantiles for prediction intervals.
        logger: logger
    Returns:
        preds: Predictions.
        interval: Prediction intervals.
        error: error message, if any
    """
    strategy = "Generic"
    error = None
    preds = np.array([])
    interval = np.array([])
    try:
        # Make predictions using the regressor and the specified quantiles
        predict = regressor.predict(X_transformed, quantiles)

        # Extract the median prediction and the prediction intervals
        preds = predict[:, 1]
        interval = predict[:, [0, 2]]
    except Exception as e:
        # If an error occurs, capture the error message
        error = str(e)

    # Return the predictions, intervals, and any error message
    return strategy, preds, interval, error


def check_in_interval(
    categorical_features: List[str],
    numerical_features: List[str],
    model: Callable,
    target: str,
    parameters: Optional[Dict[str, Union[int, float]]] = None,
    quantiles: List[float] = [0.05, 0.5, 0.95],
) -> pd.DataFrame:
    """
    Checks if actual values are within prediction intervals.
    Args:
        categorical_features: List of categorical features.
        numerical_features: List of numerical features.
        model: Model to train.
        target: Target variable.
        parameters: parameters for the regressor model.
        quantiles: Quantiles for prediction intervals. Default is [0.05, 0.5, 0.95].
    Returns:
        result: DataFrame with actual values, predictions, prediction intervals, and whether actual values are within prediction intervals.
    """
    X_train, y_train, X_test, y_test = read_data()
    X_train, X_test, data_transform_pipeline = fit_transform(X_train, X_test, categorical_features, numerical_features)

    regressor = train(
        X_train,
        y_train,
        model,
        target,
        parameters,
        scoring=mean_absolute_percentage_error,
    )
    strategy, preds, interval, error = test(X_test, y_test, regressor, data_transform_pipeline, quantiles=quantiles)
    preds = pd.DataFrame(preds, columns=["prediction"])
    interval = pd.DataFrame(interval, columns=["low", "high"])

    result = pd.concat([y_test.reset_index(drop=True), preds.reset_index(drop=True), interval.reset_index(drop=True)], axis=1)

    result.index = y_test.index
    result["in_range"] = result.apply(lambda row: 1 if row["low"] <= row[0] <= row["high"] else 0, axis=1)
    in_range = result.in_range.sum() / len(result.in_range)

    print("{} quantiles: {} in range".format(quantiles, in_range))
    return result


# finetune RandomForrestRegressor
param_grid = {
    "n_estimators": [50],
    "max_depth": [20],
    "min_samples_split": [2, 4, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
# ExtraTreesRegressor
param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [5, 25, 50, 100],
}


def finetune_model(
    categorical_features: List[str],
    numerical_features: List[str],
    model: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    verbose: int = 1,
    n_jobs: int = -1,
    scoring: str = "neg_mean_absolute_error",
) -> BaseEstimator:
    """
    Finetune the given machine learning model using grid search.
    Args:
    - categorical_features: List of categorical feature names.
    - numerical_features: List of numerical feature names.
    - model: The machine learning model to be finetuned.
    - param_grid: The parameter grid to search over.
    - verbose: The verbosity level for grid search.
    - n_jobs: The number of jobs to run in parallel for grid search.
    - scoring: The scoring method to use for evaluating the model.
    Returns:
    - The best estimator found by grid search.
    """
    X_train, y_train, X_test, y_test = read_data()
    X_train, X_test, data_transform_pipeline = fit_transform(X_train, X_test, categorical_features, numerical_features)
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=5, n_jobs=n_jobs, verbose=verbose)
    grid.fit(X_train, y_train)
    print("Best parameters: {}".format(grid.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best estimator:\n{}".format(grid.best_estimator_))
    preds = grid.predict(X_train)
    score = mean_absolute_percentage_error(y_train, preds)
    print("{} Train error: {}".format(model, score))
    preds = grid.predict(X_test)
    score = mean_absolute_percentage_error(y_test, preds)
    print("{} Test error: {}".format(model, score))
    return grid.best_estimator_


models = [
    {"name": "Random Forest", "model": RandomForestRegressor},
    {"name": "ExtraTreesRegressor", "model": ExtraTreesRegressor},
]


def compare_models(
    categorical_features: List[str],
    numerical_features: List[str],
    models: List[Dict[str, Callable]],
    target: str,
    parameters: Optional[Dict[str, Union[int, float]]] = None,
    scoring: Any = mean_absolute_percentage_error,
) -> BaseEstimator:
    """
    Compare several different machine learning models to find the best one.
    Args:
    - categorical_features: List of categorical feature names.
    - numerical_features: List of numerical feature names.
    - models: A list of dictionaries containing model names and model classes.
    - target: The target variable name.
    - parameters: Parameters to set for each model.
    - scoring: The scoring function to use for evaluating the models.
    Returns:
    - The best model class after comparison.
    """
    X_train, y_train, X_test, y_test = read_data()
    X_train, X_test, data_transform_pipeline = fit_transform(X_train, X_test, categorical_features, numerical_features)
    best_model = None
    best_score = np.inf
    for model in models:
        if parameters:
            model["model"].set_params(**parameters)

        regressor = train(
            X_train,
            y_train,
            model,
            target,
            parameters,
            scoring=mean_absolute_percentage_error,
        )
        preds = regressor.predict(X_test)
        score = scoring(y_test, preds)
        if score < best_score:
            best_score = score
            best_model = model
        print("{} Test error: {}".format(model["name"], score))
    print("\nBest Model: {}, Error: {}".format(best_model["name"], best_score))
    return best_model["model"]

