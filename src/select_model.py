"""
Model selection with the given model list based on MAE
The best 2 models will be lead to GridSearchCV to find the best hyperparameters
"""

from sklearn.model_selection import cross_val_score, GridSearchCV
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
import pandas as pd
from typing import Callable, List
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.svm import SVR


# coutinue with XGboost
# finish dicide_model function
# ------try baysian optimization
# visualization

params_model_selection = {
    "scoring": "neg_mean_absolute_percentage_error",
    "cv": 3,
    "n_jobs": -1,
}

params_finetune = {
    "scoring": "neg_mean_absolute_percentage_error",
    "n_jobs": -1,
    "cv": 5,
    "verbose": 1,
}

# a dictionart of model and its hyperparameters
# the first value of hyperparameters will be used for model selection
model_params_dict = {
    "RandomForestQuantileRegressor": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 50, 100],
        "n_jobs": [-1],
        "random_state": [42],
        "ccp_alpha": [0.0, 0.1, 1.0],
        "min_impurity_decrease": [0.0, 0.01, 0.1],
    },
    "ExtraTreesQuantileRegressor": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 50, 100],
        "n_jobs": [-1],
        "random_state": [42],
        "ccp_alpha": [0.0, 0.1, 1.0],
        "min_impurity_decrease": [0.0, 0.01, 0.1],
    },
    "XGBRegressor": {},
    "SVR": {
        "kernel": ["rbf", "poly"],
        "gamma": ["scale", 0.1, 1],
        "C": [0.1, 1, 10],
        "epsilon": [0.1, 0.2, 0.5],
    },
}


def decide_model(
    model_params_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params_model_selection: dict,
    params_finetune: dict,
) -> BaseEstimator:
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
    best_models = select_model(model, X_train, y_train)
    best_model = finetune_model(
        best_models[0], X_train, y_train, X_test, y_test, params_finetune
    )
    return best_model


def select_model(
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
    model_scores.sort(key=lambda x: x[1])
    print("Model scores: {}".format(model_scores))
    best_models = [model for model, _ in model_scores[:2]]
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
) -> BaseEstimator:
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
    print("Best parameters: {}".format(grid.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best estimator:\n{}".format(grid.best_estimator_))
    final_model = grid.best_estimator_
    preds = final_model.predict(X_test)
    score = mean_absolute_percentage_error(y_test, preds)
    print("{} Test error: {}".format(model, score))
    return grid.best_estimator_  # Return the best estimator found by grid search


# # finetune RandomForrestRegressor
# param_grid = {
#     "n_estimators": [50],
#     "max_depth": [20],
#     "min_samples_split": [2, 4, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
# }
# # ExtraTreesRegressor
# param_grid = {
#     "n_estimators": [50, 100, 200, 500],
#     "max_depth": [5, 25, 50, 100],
# }
