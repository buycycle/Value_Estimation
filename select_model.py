"""
Model selection with the given model list based on MAE
The best 2 models will be lead to GridSearchCV to find the best hyperparameters
"""

from sklearn.model_selection import cross_val_score, GridSearchCV
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
import pandas as pd
from datetime import datetime   
import json
from typing import Callable, List, Dict, Tuple
from sklearn.base import BaseEstimator
from src.data import create_data, fit_transform, create_data_model
from src.driver import (
    main_query,
    main_query_dtype,
    numerical_features,
    categorical_features,
    target,
)
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
import os
import time
import pickle
from joblib import dump, load

import logging

logging.basicConfig(level=logging.INFO)
today_date = datetime.now().strftime("%Y-%m-%d")

# Define the time limit for the files (7 days)
current_time = time.time()
time_limit = 7 * 24 * 60 * 60

path = "data"

params_model_selection = {
    "scoring": "neg_mean_absolute_percentage_error",
    "cv": 3,
    "n_jobs": -1,
}

params_finetune = {
    "scoring": "neg_mean_absolute_percentage_error",
    "n_jobs": -1,
    "cv": 2,
    "verbose": 1,
}

# a dictionart of model and its hyperparameters
# the first value of hyperparameters will be used for model selection
model_params_dict = {
    RandomForestQuantileRegressor: {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 50, 100],
        "n_jobs": [-1],
        "random_state": [42],
        "ccp_alpha": [0.0, 0.1, 1.0],
        "min_impurity_decrease": [0.0, 0.01, 0.1],
    },
    ExtraTreesQuantileRegressor: {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 50, 100],
        "n_jobs": [-1],
        "random_state": [42],
        "ccp_alpha": [0.0, 0.1, 1.0],
        "min_impurity_decrease": [0.0, 0.01, 0.1],
    },
    XGBRegressor: {
        "n_estimators": [100, 200, 500],
        "max_depth": [25, 50, 100],
        "learning_rate": [0.01, 0.1, 0.5],
        "n_jobs": [-1],
    },
    SVR: {
        "kernel": ["rbf", "poly"],
        "gamma": ["scale", 0.1, 1],
        "C": [0.1, 1, 10],
        "epsilon": [0.1, 0.2, 0.5],
    },
}


class ModelSelector:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = create_data(
            main_query,
            main_query_dtype,
            numerical_features,
            categorical_features,
            target,
            0.12,
        )

        self.X_train, self.X_test, _ = fit_transform(self.X_train, self.X_test)
        logging.info("data and data-transform-pipeline are created")

    def apply_model(self, model_params_dict, params_model_selection, params_finetune):
        self.load_data()
        best_model, score, params = model_selection(
            self.X_train, self.y_train, self.X_test, self.y_test, model_params_dict, params_model_selection, params_finetune)

        ModelClass = type(best_model)
        create_data_model(
            path=path,
            main_query=main_query,
            main_query_dtype=main_query_dtype,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            model=ModelClass,
            target=target,
            test_size=0.12,
            parameters=params,
        )

        print("end of model selection")


def select_models(
    models: List[BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = mean_absolute_percentage_error,
    cv: int = 3,
    n_jobs: int = -1,
) -> List[BaseEstimator]:
    """
    Select the best 2 models based on MAE
    Args:
    - models: The list of models to be evaluated.
    - X_train: The input data.
    - y_train: The target data.
    - scoring: The scoring method to use for evaluating the models.
    - cv: The number of cross-validation folds.
    - n_jobs: The number of jobs to run in parallel.
    Returns:
    - The best 2 models based on MAE.
    """
    scores = []
    for model in models:
        score = cross_val_score(
            model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs
        ).mean()
        scores.append(score)
    model_scores = list(zip(models, scores))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    best_models = [model for model, _ in model_scores[:2]]
    logging.info("Model scores: {}".format(model_scores))

    # Continue writing in append mode, adding the best models
    with open(f"{path}/model_scores_{today_date}.txt", "w") as file:
        file.write("Result of models: {}\n".format(model_scores))
    return best_models


def finetune_model(
    model: Callable,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: dict,
    scoring: str = "neg_mean_absolute_percentage_error",
    n_jobs: int = -1,
    cv: int = 5,
    verbose: int = 1,
) -> Tuple[BaseEstimator, float, Dict]:
    """
    Finetune the given machine learning model using grid search.
    Args:
    - model: The machine learning model to be finetuned.
    - X_train: The training data.
    - y_train: The training labels.
    - X_test: The test data.
    - y_test: The test labels.
    - param_grid: The parameter grid to search over.
    - scoring: The scoring method to use for evaluating the model.
    - n_jobs: The number of jobs to run in parallel for grid search.
    - cv: The number of cross-validation folds to use for grid search.
    - verbose: The verbosity level for grid search.
    Returns:
    - The best estimator found by grid search.
    """
    grid = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        verbose=verbose,
    )
    grid.fit(X_train, y_train)

    final_model = grid.best_estimator_
    preds = final_model.predict(X_test)
    score = mean_absolute_percentage_error(y_test, preds)
    logging.info(
        f"Model {final_model} with best cv score {grid.best_score_} and test scores: {score}"
    )
    all_results = grid.cv_results_
    with open(f"{path}/model_scores_{today_date}.txt", "a") as file:
        file.write(f"Result of finetuning: {model.__class__.__name__}\n")
        for key, value in all_results.items():
            file.write(f"{key}: {value}\n")


    # Return the best estimator found by grid search
    return grid.best_estimator_, grid.best_score_, grid.best_params_


def model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_params_dict: dict,
    params_model_selection: dict,
    params_finetune: dict,
) -> Tuple[BaseEstimator, float, Dict]:
    """
    Decide the best model based on MAE
    First step, choose the best 2 models from a list of model
    Second step, finetune the 2 best model with the best hyperparameters
    Final step, return the best model
    Args:
    - model_and_params: The dict of model and its hyperparameters.
    - X_train: The input data.
    - y_train: The target data.
    - X_test: The test data.
    - y_test: The test data.
    - params_model_selection: The parameter for model selection.
    - params_finetune: The parameter for model finetuning.
    Returns:
    - The best model based on MAE.
    """
    # Run select_models for each model with the first set of parameters
    models = []
    for model_class, params in model_params_dict.items():
        default_params = {param: values[0] for param, values in params.items()}

        # Initialize the model with the first set of parameters
        model = model_class(**default_params)
        models.append(model)

    top_models = select_models(models, X_train, y_train, **params_model_selection)

    # Run finetune_model for best 2  model with the full set of parameters
    finetuned_results = []
    for model in top_models:
        model_name = model.__class__.__name__
        model_params = model_params_dict[type(model)]
        finetuned_result = finetune_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_params,
            **params_finetune,
        )
        finetuned_results.append(finetuned_result)

    # Return the best model
    best_result = min(finetuned_results, key=lambda x: abs(x[1]))
    logging.info("Best model: {}".format(best_result))

    return best_result


model_selector = ModelSelector()
model_selector.apply_model(model_params_dict, params_model_selection, params_finetune)
