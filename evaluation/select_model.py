"""
Model selection with the given model list based on MAE
The best 2 models will be lead to GridSearchCV to find the best hyperparameters
"""

from sklearn.model_selection import cross_val_score, GridSearchCV
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
import pandas as pd
from typing import Callable

# sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
# class quantile_forest.RandomForestQuantileRegressor(n_estimators=100, *, default_quantiles=0.5, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)


def gscv_model(
    model: Callable,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: int = 1,
    n_jobs: int = -1,
    scoring: str = "mean_absolute_percentage_error",
) -> BaseEstimator:
    """
    Finetune the given machine learning model using grid search.
    Args:
    - model: The machine learning model to be finetuned.
    - param_grid: The parameter grid to search over.
    - X_train: The training data.
    - y_train: The training labels.
    - X_test: The test data.
    - y_test: The test labels.
    - verbose: The verbosity level for grid search.
    - n_jobs: The number of jobs to run in parallel for grid search.
    - scoring: The scoring method to use for evaluating the model.
    Returns:
    - The best estimator found by grid search.
    """
    grid = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
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


# def finetune_model(
#     categorical_features: List[str],
#     numerical_features: List[str],
#     model: BaseEstimator,
#     param_grid: Dict[str, List[Any]],
#     verbose: int = 1,
#     n_jobs: int = -1,
#     scoring: str = "neg_mean_absolute_error",
# ) -> BaseEstimator:
#     """
#     Finetune the given machine learning model using grid search.
#     Args:
#     - categorical_features: List of categorical feature names.
#     - numerical_features: List of numerical feature names.
#     - model: The machine learning model to be finetuned.
#     - param_grid: The parameter grid to search over.
#     - verbose: The verbosity level for grid search.
#     - n_jobs: The number of jobs to run in parallel for grid search.
#     - scoring: The scoring method to use for evaluating the model.
#     Returns:
#     - The best estimator found by grid search.
#     """
#     X_train, y_train, X_test, y_test = read_data()
#     X_train, X_test, data_transform_pipeline = fit_transform(
#         X_train, X_test, categorical_features, numerical_features
#     )
#     grid = GridSearchCV(
#         model, param_grid, scoring=scoring, cv=5, n_jobs=n_jobs, verbose=verbose
#     )
#     grid.fit(X_train, y_train)
#     print("Best parameters: {}".format(grid.best_params_))
#     print("Best cross-validation score: {:.2f}".format(grid.best_score_))
#     print("Best estimator:\n{}".format(grid.best_estimator_))
#     preds = grid.predict(X_train)
#     score = mean_absolute_percentage_error(y_train, preds)
#     print("{} Train error: {}".format(model, score))
#     preds = grid.predict(X_test)
#     score = mean_absolute_percentage_error(y_test, preds)
#     print("{} Test error: {}".format(model, score))
#     return grid.best_estimator_


# example code for nested cross-validation
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups

# Load text data
data = fetch_20newsgroups(subset="all", categories=["alt.atheism", "sci.space"])
X, y = data.data, data.target

# Define a pipeline with TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Define hyperparameter grid
param_grid = {
    "tfidfvectorizer__max_df": [0.75, 1.0],
    "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
    "multinomialnb__alpha": [0.01, 0.1, 1],
}

# Define the inner and outer cross-validation strategies
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define GridSearchCV with inner cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="accuracy")

# Perform nested cross-validation with outer cross-validation
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)

print("Nested cross-validation scores:", nested_scores)
print("Mean nested cross-validation accuracy:", nested_scores.mean())
