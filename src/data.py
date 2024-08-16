import os
import numpy as np
import pandas as pd
import threading  # for data read lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
import category_encoders as ce
from joblib import dump, load
from buycycle.data import sql_db_read, DataStoreBase
from src.price import train
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
from src.driver import (
    cumulative_inflation_df,
    target,
    msrp_min,
    msrp_max,
    target_min,
    target_max,
    categorical_features,
    numerical_features,
    main_query_dtype,
)

def get_data(
    main_query: str,
    main_query_dtype: str,
    index_col: str = "id",
    config_paths: str = "config/config.ini",
) -> pd.DataFrame:
    """
    Fetches data from SQL database.
    Args:
        main_query: SQL query for main data.
        main_query_dtype: Data type for main query.
        index_col: Index column for DataFrame. Default is 'id'.
        config_paths: Path to configuration file. Default is 'config/config.ini'.
    Returns:
        DataFrame: Main data.
    """
    df = sql_db_read(
        query=main_query,
        DB="DB_BIKES",
        config_paths=config_paths,
        dtype=main_query_dtype,
        index_col=index_col,
    )
    return df


def clean_data(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    target: str = target,
) -> pd.DataFrame:
    """
    Cleans data by removing outliers and unnecessary data.
    Args:
        df: DataFrame to clean.
        numerical_features: List of numerical feature names.
        categorical_features: List of categorical feature names.
        target: Target column. Default is 'sales_price'.
        iqr_limit: IQR limit for outlier detection. Default is 2.
    Returns:
        DataFrame: Cleaned data.
    """
    # keep the order of features, which is the same with the api
    # contain bike_created_at for oversampling and will be drop afterwards
    column_order = [
        col
        for col in categorical_features
        + numerical_features
        + [target]
        + ["bike_created_at", "bike_year"]
        if col not in ["template_ud", "bike_created_at_month_sin", "bike_created_at_month_cos", "bike_age"]
    ]
    df = df[column_order]

    # exclude data with the low/high price and msrp
    df = df[(df[target] > target_min) & (df[target] < target_max)]
    df = df[
        (df["msrp"].notnull()) & (df["msrp"] >= msrp_min) & (df["msrp"] <= msrp_max)
    ]
    # exclude data withreally low price(pontential fake bike)
    df = df[df[target] > df["msrp"] * 0.3]

    # remove duplicates
    df = df.loc[~df.index.duplicated(keep="last")]

    # adjust sales_price in bike_created_at_year to current value
    df = pd.merge(
        df,
        cumulative_inflation_df,
        left_on="bike_created_at_year",
        right_on="year",
        how="left",
    )
    df[target] = df[target] * df["inflation_factor"]
    df.drop(["year", "inflation_factor"], axis=1, inplace=True)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for both training, testing data, also the request
    Args:
        df: DataFrame to feature engineering.
    Returns:
        DataFrame: Transformed data.
    """
    # set year and month to current time if not given
    current_year = datetime.now().year
    current_month = datetime.now().month

    df["bike_created_at_year"].fillna(current_year, inplace=True)
    df["bike_created_at_month"].fillna(current_month, inplace=True)

    # replace bike_created_at_month with a sinusoidal transformation
    df["bike_created_at_month_sin"] = np.sin(
        2 * np.pi * df["bike_created_at_month"] / 12
    )
    df["bike_created_at_month_cos"] = np.cos(
        2 * np.pi * df["bike_created_at_month"] / 12
    )
    # set missing/problematic bike_year to 2018 before doing inflation calculate
    df["bike_year"] = df["bike_year"].apply(
        lambda x: 2018 if pd.isnull(x) or x < 1900 or x > current_year else x
    )

    # adjust msrp to bike_year according to the EU inflation rate, backtracking only to 2020
    df["merge_year"] = df["bike_year"].apply(lambda x: 2020 if x < 2020 else x)
    df = pd.merge(
        df, cumulative_inflation_df, left_on="merge_year", right_on="year", how="left"
    )
    df["msrp"] = df["msrp"] * df["inflation_factor"]
    # apply a logarithmic transformation since 'msrp' distribution is right-skewed
    df["msrp"] = np.log(df["msrp"] + 1)
    df.drop(["merge_year", "year", "inflation_factor"], axis=1, inplace=True)

    # create bike age from bike_year
    df["bike_age"] = df["bike_created_at_year"] - df["bike_year"]

    # drop unused columns
    df.drop(["bike_year"], axis=1, inplace=True)

    # replace mileage_code with number, treated as numerical feature
    df["mileage_code"] = (
        df["mileage_code"]
        .replace(
            {
                "more_than_10000": 5,
                "3000_10000": 4,
                "500_3000": 3,
                "less_than_500": 2,
                "0km": 1,
            }
        )
        .fillna(-1)
    )
    # convert condition_code from string to integer, treated as numerical feature
    df["condition_code"] = pd.to_numeric(df["condition_code"], errors="coerce")

    # Replace 'None' with np.nan across all columns in the DataFrame
    # in categorical/string columns, np.nan will be imputed as missing data, otherwise the None and missing data will be treated as different
    df.replace("None", np.nan, inplace=True)
    print(f"after featured: {df.columns}")
    return df


def train_test_split_date(df: pd.DataFrame, target: str, test_size: float = 0.12):
    """
    Splits data into training and test sets based on a date cutoff.
    Args:
        df: DataFrame to split, including target column
        target: Target column.
        size: the size of test data, default = 0.12
    Returns:
        X_train, y_train, X_test, y_test: Training and test sets.
    """
    X = df.drop([target], axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=21
    )

    return X_train, y_train, X_test, y_test


def oversample_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target: str,
    days: int = 60,
    duplication_factor: int = 3,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Oversample the last 30 days of data with duplication_factor

    Parameters:
    - X_train: pd.DataFrame, the features of the training data.
    - y_train: pd.Series, the target variable of the training data.
    - target: str, the name of the target column.
    - days: int, the number of days to consider for oversampling.
    - duplication_factor: int, the factor by which to duplicate the recent data.

    Returns:
    - X_train_oversampled: pd.DataFrame, the oversampled features of the training data.
    - y_train_oversampled: pd.Series, the oversampled target variable of the training data.
    """
    today = datetime.now()
    threshold_date = today - timedelta(days=days)

    train_df = X_train.copy()
    train_df[target] = y_train.values

    # Filter recent data
    train_df["bike_created_at"] = pd.to_datetime(train_df["bike_created_at"])
    recent_data = train_df[train_df["bike_created_at"] >= threshold_date]

    # Duplicate recent data
    duplicated_recent_data = pd.concat(
        [recent_data] * duplication_factor, ignore_index=True
    )

    # Combine original data with duplicated recent data
    oversampled_train_df = pd.concat(
        [train_df, duplicated_recent_data], ignore_index=True
    )

    # Shuffle the dataset to mix duplicated and original data
    oversampled_train_df = oversampled_train_df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    X_train_oversampled = oversampled_train_df.drop(columns=[target], axis=1)
    y_train_oversampled = oversampled_train_df[target]  # Target column

    return X_train_oversampled, y_train_oversampled


def create_data(
    query: str,
    query_dtype: str,
    numerical_features: List[str],
    categorical_features: List[str],
    target: str = target,
    test_size: float = 0.12,
):
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
    df = get_data(query, query_dtype)
    df = clean_data(df, numerical_features, categorical_features, target=target).sample(frac=1)
    df = feature_engineering(df)
    X_train, y_train, X_test, y_test = train_test_split_date(
        df, target, test_size=test_size
    )
    X_train, y_train = oversample_data(X_train, y_train, target, 60, 3)
    # drop bike_created_at column, which is not a feature in model
    X_train = X_train.drop(columns=["bike_created_at"], axis=1)
    X_test = X_test.drop(columns=["bike_created_at"], axis=1)

    return X_train, X_test, y_train, y_test


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


# class ModeImputer(BaseEstimator, TransformerMixin):
#     """
#     Class to fill NA values with mode
#     """

#     def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "ModeImputer":
#         """
#         Fit the Imputer on X.
#         Parameters: X : DataFrame (input data), y : DataFrame (optional, for API consistency)
#         Returns: self : Imputer (fitted instance)
#         """
#         self.most_frequent_ = pd.DataFrame(X).mode().iloc[0]
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         """
#         Fill NA values with mode.
#         Parameters: X : DataFrame (input data)
#         Returns: DataFrame (transformed data)
#         """
#         return pd.DataFrame(X).fillna(self.most_frequent_)


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

    def __init__(
        self,
        categorical_features,
        numerical_features,
        n_estimators=100,
        max_depth=5,
        random_state=0,
    ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_features}
        self.imp_num = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            ),
            initial_strategy="mean",
            max_iter=10,
            random_state=self.random_state,
        )
        self.imp_cat = IterativeImputer(
            estimator=RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
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
            transformed_data.loc[valid_data.index] = self.label_encoders[col].transform(
                valid_data
            )
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
            X[self.categorical_features] = X[self.categorical_features].apply(
                pd.to_numeric, errors="coerce"
            )
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
            X_transformed[self.numerical_features] = self.imp_num.transform(
                X[self.numerical_features]
            )
        # Impute categorical features
        if self.categorical_features:
            # Convert categorical columns to numeric for imputation
            X_cat_numeric = X[self.categorical_features].apply(
                lambda col: pd.to_numeric(col, errors="coerce")
            )
            X_cat_imputed = self.imp_cat.transform(X_cat_numeric)
            # Decode the categorical features back to original labels
            for idx, col in enumerate(self.categorical_features):
                # Round imputed values to the nearest integer
                imputed_labels = np.round(X_cat_imputed[:, idx])
                # Get the numeric representation of the known classes
                known_labels = np.arange(len(self.label_encoders[col].classes_))
                # Clip the imputed labels to the range of known numeric labels
                imputed_labels = np.clip(
                    imputed_labels, known_labels.min(), known_labels.max()
                )
                # Inverse transform the labels to original categories
                X_transformed[col] = self.label_encoders[col].inverse_transform(
                    imputed_labels.astype(int)
                )
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
            ce.OneHotEncoder(
                cols=self.categorical_features,
                handle_unknown="indicator",
                use_cat_names=True,
            )
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

    def __init__(
        self,
        scaler_name: str,
        numerical_features: Optional[List[str]] = None,
        scaler_params: Optional[Dict] = None,
    ):
        """
        Initialize Scaler.
        Parameters: numerical_features : list of str (optional, default scales all)
        """
        self.scaler_params = scaler_params
        self.numerical_features = numerical_features
        if scaler_name == "MinMaxScaler":
            self.scaler_ = (
                MinMaxScaler(**scaler_params) if scaler_params else MinMaxScaler()
            )
        elif scaler_name == "RobustScaler":
            self.scaler_ = RobustScaler()
        else:
            raise ValueError(
                f"Invalid scaler_name: {scaler_name}. Expected 'MinMaxScaler' or 'RobustScaler'."
            )

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "Scaler":
        """
        Fit Scaler on X.
        Parameters: X : DataFrame (input data), y : DataFrame (optional, for API consistency)
        Returns: self : Scaler (fitted instance)
        """
        (
            self.scaler_.fit(X[self.numerical_features])
            if self.numerical_features
            else self.scaler_.fit(X)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        Parameters: X : DataFrame (input data)
        Returns: DataFrame (transformed data)
        """
        X.loc[:, self.numerical_features] = (
            self.scaler_.transform(X[self.numerical_features])
            if self.numerical_features
            else self.scaler_.transform(X)
        )
        return X


def create_data_transform_pipeline(
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
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

    # Scaler for numerical features except msrp
    numerical_features_reduced = numerical_features.copy()
    numerical_features_reduced.remove("msrp")
    numerical_scaler = Scaler("MinMaxScaler", numerical_features_reduced)

    # Custom scaler for msrp
    msrp_robus_scaler = Scaler("RobustScaler", ["msrp"])
    msrp_scaler = Scaler(
        "MinMaxScaler", ["msrp"], scaler_params={"feature_range": (0, msrp_max)}
    )

    # Create the pipeline with the custom imputer, one-hot encoder, and scaler
    data_transform_pipeline = Pipeline(
        steps=[
            ("impute", miss_forest_imputer),
            ("dummies", categorical_encoder),
            ("scale", numerical_scaler),
            ("msrp_robus_scaler", msrp_robus_scaler),
            ("msrp_scaler", msrp_scaler),
        ]
    )
    return data_transform_pipeline


def fit_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """
    Fit and transform the data using the pipeline.
    Args:
    ----------
    X_train : DataFrame
        Training data.
    X_test : DataFramecreate_data_transform_pipeline
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
    print("categorical_features: {}".format(categorical_features))
    print("numerical_features: {}".format(numerical_features))

    data_transform_pipeline = create_data_transform_pipeline(
        categorical_features, numerical_features
    )
    X_train = data_transform_pipeline.fit_transform(X_train)
    X_test = data_transform_pipeline.transform(X_test)

    return X_train, X_test, data_transform_pipeline


def write_model_pipeline(
    regressor: Callable, data_transform_pipeline: Callable, path: str
) -> None:
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
    test_size: float = 0.12,
    parameters: Optional[Dict[str, Union[int, float]]] = None,
) -> None:
    """
    Create data and model
    write the data, model and datapipeline to a given path.
    Args:
    - path: The path where the model and pipeline will be written.
    - main_query: The main query to create data.
    - main_query_dtype: Data types for the main query.
    - numerical_features: List of numerical feature names.
    - categorical_features: List of categorical feature names.
    - model: The machine learning model to be trained.
    - target: The target variable name.
    - months: The number of months to consider in the data.
    - parameters: Optional dictionary of hyperparameters for the model.
    Side effects:
    - Writes the trained model and data transformation pipeline to the specified path.
    Returns:
    - None
    """
    X_train, X_test, y_train, y_test = create_data(
        main_query,
        main_query_dtype,
        numerical_features,
        categorical_features,
        target,
        test_size,
    )

    X_train, X_test, data_transform_pipeline = fit_transform(X_train, X_test)

    regressor = train(
        X_train,
        y_train,
        model,
        parameters,
        scoring=mean_absolute_percentage_error,
    )

    X_train.to_pickle(os.path.join(path, "X_train.pkl"))
    y_train.to_pickle(os.path.join(path, "y_train.pkl"))
    X_test.to_pickle(os.path.join(path, "X_test.pkl"))
    y_test.to_pickle(os.path.join(path, "y_test.pkl"))
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
        return {
            "regressor_info": str(self.regressor),
            "data_transform_pipeline": str(self.data_transform_pipeline),
        }
