
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
from src.price import train
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor

import threading # for data read lock

def get_data(
    main_query: str, main_query_dtype: str, index_col: str = "sales_price", config_paths: str = "config/config.ini"
) -> pd.DataFrame:
    """
    Fetches data from SQL database.
    Args:
        main_query: SQL query for main data.
        main_query_dtype: Data type for main query.
        index_col: Index column for DataFrame. Default is 'sales_price'.
        config_paths: Path to configuration file. Default is 'config/config.ini'.
    Returns:
        DataFrame: Main data.
    """
    df = sql_db_read(query=main_query, DB="DB_BIKES", config_paths=config_paths, dtype=main_query_dtype, index_col=index_col)
    return df


def clean_data(
    df: pd.DataFrame, numerical_features: List[str], target: str = "sales_price", iqr_limit: float = 3
) -> pd.DataFrame:
    """
    Cleans data by removing outliers and unnecessary data.
    Args:
        df: DataFrame to clean.
        numerical_features: List of numerical feature names.
        target: Target column. Default is 'sales_price'.
        iqr_limit: IQR limit for outlier detection. Default is 2.
    Returns:
        DataFrame: Cleaned data.
    """
    # remove custom template idf and where target = NA
    df = df[df["template_id"] != 79204].dropna(subset=[target])
    df = df[df[target] > 400]
    df = df[df[target] < 15000]
    for col in numerical_features:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[~((df[col] < (q1 - iqr_limit * iqr)) | (df[col] > (q3 + iqr_limit * iqr)))]
    # remove duplicates
    df = df.loc[~df.index.duplicated(keep="last")]
    return df

def feature_engineering(df: pd.DataFrame, numerical_features: List[str], categorical_features: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # replace bike_created_at_month with a sinusoidal transformation
    df["bike_created_at_month_sin"] = np.sin(2 * np.pi * df["bike_created_at_month"] / 12)
    # create bike age from bike_year
    df["bike_age"] = pd.to_datetime("today").year - df["bike_year"]

    # add bike_created_at_month_sin to numerical features
    numerical_features.append("bike_created_at_month_sin")
    numerical_features.append("bike_age")

    return df, numerical_features, categorical_features




def train_test_split_date(df, numerical_features, categorical_features, target, months):
    """
    Splits data into training and test sets based on a date cutoff.
    Args:
        df: DataFrame to split.
        target: Target column.
        months: Number of months before current date to use as cutoff.
    Returns:
        X_train, y_train, X_test, y_test: Training and test sets.
    """
    df["bike_created_at"] = pd.to_datetime(
        df["bike_created_at_year"].astype(str) + df["bike_created_at_month"].astype(str), format="%Y%m"
    )
    cutoff_date = pd.to_datetime("today") - pd.DateOffset(months=months)
    train, test = df[df["bike_created_at"] <= cutoff_date], df[df["bike_created_at"] > cutoff_date]
    X_train, y_train = train.drop([target, "bike_created_at", "bike_created_at_month", "bike_year"], axis=1), train[target]
    X_test, y_test = test.drop([target, "bike_created_at", "bike_created_at_month", "bike_year"], axis=1), test[target]
    # remove bike_created at month from numerical features
    if "bike_created_at_month" in numerical_features:
        numerical_features.remove("bike_created_at_month")
    if "bike_year" in numerical_features:
        numerical_features.remove("bike_year")

    return X_train, y_train, X_test, y_test, numerical_features, categorical_features



def create_data(query: str, query_dtype: str, numerical_features: List[str], categorical_features: List[str], target: str, months: int, path: str = "data/"):
    """
    Fetches, cleans, splits, and saves data.
    Args:
        query: SQL query for main data.
        query_dtype: Data type for main query.
        numerical_features: numerical_features.
        arget: Target column.
        months: Number of months before current date to use as cutoff.
        path: Path to save data. Default is 'data/'.
    """
    df = get_data(query, query_dtype, index_col="id")
    print(f"Dimensions of df after get_data: {df.shape}")
    df = clean_data(df, numerical_features, target=target).sample(frac=1)
    print(f"Dimensions of df after clean_data and sampling: {df.shape}")
    df, numerical_features, categorical_features = feature_engineering(df, numerical_features, categorical_features)
    print(f"Dimensions of df after feature_engineering: {df.shape}")
    X_train, y_train, X_test, y_test, numerical_features, categorical_features = train_test_split_date(df, numerical_features, categorical_features, target, months)
    X_train.to_pickle(path + "X_train.pkl")
    y_train.to_pickle(path + "y_train.pkl")
    X_test.to_pickle(path + "X_test.pkl")
    y_test.to_pickle(path + "y_test.pkl")

    return numerical_features, categorical_features


def read_data(path: str = "data/"):
    """
    Reads saved data from disk.
    Args:
        path: Path to saved data. Default is 'data/'.
    Returns:
        X_train, y_train, X_test, y_test: Training and test sets.
    """
    X_train = pd.read_pickle(path + "X_train.pkl")
    y_train = pd.read_pickle(path + "y_train.pkl")
    X_test = pd.read_pickle(path + "X_test.pkl")
    y_test = pd.read_pickle(path + "y_test.pkl")
    return X_train, y_train, X_test, y_test


class ModeImputer(BaseEstimator, TransformerMixin):
    """
    Class to fill NA values with mode
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "ModeImputer":
        """
        Fit the Imputer on X.
        Parameters: X : DataFrame (input data), y : DataFrame (optional, for API consistency)
        Returns: self : Imputer (fitted instance)
        """
        self.most_frequent_ = pd.DataFrame(X).mode().iloc[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NA values with mode.
        Parameters: X : DataFrame (input data)
        Returns: DataFrame (transformed data)
        """
        return pd.DataFrame(X).fillna(self.most_frequent_)


class MissForestImputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer class that uses RandomForestRegressor and RandomForestClassifier
    to impute missing values in numerical and categorical features, respectively.
    This is similar to the MissForest algorithm in R, which uses Random Forests
    to impute missing values iteratively.
    Parameters:
    ----------
    categorical_features : list of str
        The list of categorical feature names that need imputation.
    numerical_features : list of str
        The list of numerical feature names that need imputation.
    n_estimators : int, default=100
        The number of trees in the Random Forest used for imputing missing values.
    random_state : int, default=0
        Controls the randomness of the estimator. The same random_state ensures
        the same results each time the estimator is used.
    """

    def __init__(self, categorical_features, numerical_features, n_estimators=100, max_depth=5, random_state=0):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_features}
        self.imp_num = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
            ),
            initial_strategy="mean",
            max_iter=10,
            random_state=self.random_state,
        )
        self.imp_cat = IterativeImputer(
            estimator=RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
            ),
            initial_strategy="most_frequent",
            max_iter=10,
            random_state=self.random_state,
        )

    def fit(self, X, y=None):
        """
        Fit the imputer on the provided data.
        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to complete.
        y : ignored
            Not used, present here for API consistency by convention.
        Returns:
        -------
        self : object
            Returns self.
        """
        # Fit label encoders and transform categorical columns to numeric
        for col in self.categorical_features:
            valid_data = X[col][X[col].notnull()]
            self.label_encoders[col].fit(valid_data)
            # Store the transformed data in a temporary variable
            transformed_data = pd.Series(index=X.index)
            # Only assign transformed values to non-null indices
            transformed_data.loc[valid_data.index] = self.label_encoders[col].transform(valid_data)
            # Fill missing entries with NaN to maintain the length consistency
            transformed_data = transformed_data.reindex(X.index, fill_value=np.nan)
            # Assign the transformed data back to the DataFrame
            X[col] = transformed_data
        # Fit numerical imputer
        if self.numerical_features:
            self.imp_num.fit(X[self.numerical_features])
        # Fit categorical imputer
        if self.categorical_features:
            # Convert categorical columns to numeric for imputation
            X[self.categorical_features] = X[self.categorical_features].apply(pd.to_numeric, errors="coerce")
            self.imp_cat.fit(X[self.categorical_features])
        return self

    def transform(self, X):
        """
        Impute all missing values in X.
        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to complete.
        Returns:
        -------
        X : array-like, shape (n_samples, n_features)
            The input data with missing values imputed.
        """
        # Create a copy of the DataFrame to avoid changes to the original data
        X_transformed = X.copy()
        # Impute numerical features
        if self.numerical_features:
            X_transformed[self.numerical_features] = self.imp_num.transform(X[self.numerical_features])
        # Impute categorical features
        if self.categorical_features:
            # Convert categorical columns to numeric for imputation
            X_cat_numeric = X[self.categorical_features].apply(lambda col: pd.to_numeric(col, errors="coerce"))
            X_cat_imputed = self.imp_cat.transform(X_cat_numeric)
            # Decode the categorical features back to original labels
            for idx, col in enumerate(self.categorical_features):
                # Round imputed values to the nearest integer
                imputed_labels = np.round(X_cat_imputed[:, idx])
                # Get the numeric representation of the known classes
                known_labels = np.arange(len(self.label_encoders[col].classes_))
                # Clip the imputed labels to the range of known numeric labels
                imputed_labels = np.clip(imputed_labels, known_labels.min(), known_labels.max())
                # Inverse transform the labels to original categories
                X_transformed[col] = self.label_encoders[col].inverse_transform(imputed_labels.astype(int))
        return X_transformed


class DummyCreator(BaseEstimator, TransformerMixin):
    """
    Class for one-hot encoding
    """

    def __init__(self, categorical_features: Optional[List[str]] = None):
        """
        Initialize DummyCreator.
        Parameters: categorical_features : list of str (optional, default encodes all)
        """
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DummyCreator":
        """
        Fit DummyCreator on X.
        Parameters: X : DataFrame (input data), y : DataFrame (optional, for API consistency)
        Returns: self : DummyCreator (fitted instance)
        """
        self.encoder_ = (
            ce.OneHotEncoder(cols=self.categorical_features, handle_unknown="indicator", use_cat_names=True)
            if self.categorical_features
            else ce.OneHotEncoder(handle_unknown="indicator", use_cat_names=True)
        )
        self.encoder_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features.
        Parameters: X : DataFrame (input data)
        Returns: DataFrame (transformed data)
        """
        return self.encoder_.transform(X)


class Scaler(BaseEstimator, TransformerMixin):
    """
    Class for scaling features using MinMaxScaler
    """

    def __init__(self, numerical_features: Optional[List[str]] = None):
        """
        Initialize Scaler.
        Parameters: numerical_features : list of str (optional, default scales all)
        """
        self.scaler_ = MinMaxScaler()
        self.numerical_features = numerical_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "Scaler":
        """
        Fit Scaler on X.
        Parameters: X : DataFrame (input data), y : DataFrame (optional, for API consistency)
        Returns: self : Scaler (fitted instance)
        """
        self.scaler_.fit(X[self.numerical_features]) if self.numerical_features else self.scaler_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        Parameters: X : DataFrame (input data)
        Returns: DataFrame (transformed data)
        """
        X.loc[:, self.numerical_features] = (
            self.scaler_.transform(X[self.numerical_features]) if self.numerical_features else self.scaler_.transform(X)
        )
        return X


def create_data_transform_pipeline(
    categorical_features: Optional[List[str]] = None, numerical_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Create a pipeline for data preprocessing.
    Args:
    ----------
    categorical_features : list of str, optional
        Categorical features to process. If None, all are processed.
    numerical_features : list of str, optional
        Numerical features to process. If None, all are processed.
    Returns
    -------
    Pipeline
        Constructed pipeline.
    """
    # Custom MissForest imputer for both categorical and numerical features
    miss_forest_imputer = MissForestImputer(categorical_features, numerical_features)

    # One-hot encoding for categorical features
    categorical_encoder = DummyCreator(categorical_features)

    # Scaler for numerical features
    numerical_scaler = Scaler(numerical_features)

    # Create the pipeline with the custom imputer, one-hot encoder, and scaler
    data_transform_pipeline = Pipeline(
        steps=[
            ("impute", miss_forest_imputer),
            ("dummies", categorical_encoder),
            ("scale", numerical_scaler),
        ]
    )
    return data_transform_pipeline


def fit_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """
    Fit and transform the data using the pipeline.
    Args:
    ----------
    X_train : DataFrame
        Training data.
    X_test : DataFrame
        Testing data.
    categorical_features : list of str, optional
        Categorical features to process. If None, all are processed.
    numerical_features : list of str, optional
        Numerical features to process. If None, all are processed.
    Returns
    -------
    Tuple[DataFrame, DataFrame, Pipeline]
        Transformed training data, transformed testing data, and fitted pipeline.
    """
    if categorical_features is None:
        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    if numerical_features is None:
        numerical_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    data_transform_pipeline = create_data_transform_pipeline(categorical_features, numerical_features)
    X_train = data_transform_pipeline.fit_transform(X_train)
    X_test = data_transform_pipeline.transform(X_test)

    print(f"Model has been fit and transformed with numerical features: {numerical_features} "
          f"and categorical features: {categorical_features}")
    return X_train, X_test, data_transform_pipeline


def write_model_pipeline(regressor: Callable, data_transform_pipeline: Callable, path: str) -> None:
    """
    Save the regressor and the data transformation pipeline to a file.
    Args:
        regressor: A trained model that can be used to make predictions.
        data_transform_pipeline: A pipeline that performs data transformations.
        path: The file path where the model and pipeline should be saved.
    Returns:
        None
    """

    model_file_path = os.path.join(path, "model.joblib")
    pipeline_file_path = os.path.join(path, "pipeline.joblib")

    # Save the regressor to the model file
    dump(regressor, model_file_path)

    # Save the pipeline to the pipeline file
    dump(data_transform_pipeline, pipeline_file_path)


def read_model_pipeline(path: Optional[str] = "./data/") -> Tuple[Callable, Callable]:
    """
    Load the regressor and the data transformation pipeline from a file.
    Args:
        path: The file path from which the model and pipeline should be loaded.
    Returns:
        regressor: The loaded model that can be used to make predictions.
        data_transform_pipeline: The loaded pipeline that performs data transformations.
    """
    # Construct file paths for the model and pipeline
    model_file_path = os.path.join(path, "model.joblib")
    pipeline_file_path = os.path.join(path, "pipeline.joblib")

    # Load the regressor from the model file
    regressor = load(model_file_path)

    # Load the pipeline from the pipeline file
    data_transform_pipeline = load(pipeline_file_path)

    return regressor, data_transform_pipeline

def create_data_model(
    path: str,
    main_query: str,
    main_query_dtype: Dict[str, Any],
    numerical_features: List[str],
    categorical_features: List[str],
    model: BaseEstimator,
    target: str,
    months: int,
    parameters: Optional[Dict[str, Union[int, float]]] = None,
) -> None:
    """
    Create a data model and write the model and pipeline to a given path.
    Args:
    - path: The path where the model and pipeline will be written.
    - main_query: The main query to create data.
    - main_query_dtype: Data types for the main query.
    - categorical_features: List of categorical feature names.
    - numerical_features: List of numerical feature names.
    - model: The machine learning model to be trained.
    - target: The target variable name.
    - months: The number of months to consider in the data.
    - parameters: Optional dictionary of hyperparameters for the model.
    Side effects:
    - Writes the trained model and data transformation pipeline to the specified path.
    Returns:
    - None
    """
    numerical_features, categorical_features = create_data(main_query, main_query_dtype, numerical_features, categorical_features, target, months)
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
    write_model_pipeline(regressor, data_transform_pipeline, path)

class ModelStore(DataStoreBase):
    def __init__(self):
        super().__init__()
        self.regressor = None
        self.data_transform_pipeline = None
        self._lock = threading.Lock()  # Initialize a lock object

    def read_data(self):
        with self._lock:  # Acquire the lock before reading data
            self.regressor, self.data_transform_pipeline = read_model_pipeline()

    def get_logging_info(self):
        return {"regressor_info": str(self.regressor), "data_transform_pipeline": str(self.data_transform_pipeline)}
