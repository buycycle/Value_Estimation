from typing import Any, List
import pandas as pd
import numpy as np


def interveave(list1: list, list2: list) -> list:
    """interveave two lists"""

    return [item for x in zip(list1, list2) for item in x] + (
        list2[len(list1) :] if len(list2) > len(list1) else list1[len(list2) :]
    )


def get_field_value(dataframe: pd.DataFrame, field: str, default_value: Any, dtype: type = int) -> Any:
    """
    Function to get the field value from a DataFrame or return a default value if any of these contitions are met;
    field does not exist, has a null value, a string in the na list.

    Parameters:
    dataframe (pd.DataFrame): DataFrame to extract the field value from.
    field (str): Field/column name to extract from the dataframe.
    default_value (Any): Default value to return if the field does not exist or its value is null.
    dtype (type): Datatype to convert the field value to. Default is int.

    Returns:
    Any: The field value from the DataFrame converted to the specified datatype.
    If the field does not exist or its value is null, the default value is returned.
    """
    na_values = ['N/A', 'NA', 'na', 'n/a', 'NaN', 'nan', None, 'None', 'NONE', '']
    if field not in dataframe or dataframe[field].iloc[0] in na_values:
        return dtype(default_value)
    else:
        return dtype(dataframe[field].iloc[0])

def construct_input_df(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Constructs a DataFrame based on the specified columns. If a column is present in the input DataFrame,
    it is copied to the resulting DataFrame. If a column is missing, it is filled with pd.NA values.
    Parameters:
    - df (pd.DataFrame): The input DataFrame to check for column availability.
    - columns (List[str]): A list of column names that the resulting DataFrame should contain.
    Returns:
    - pd.DataFrame: A DataFrame with the specified columns, containing original data for present columns
      and pd.NA for missing columns.
    """
    # Create a new DataFrame with the same index as the input DataFrame
    result_df = pd.DataFrame(index=df.index)

    # Iterate over the list of columns
    for column in columns:
        # Check if the column is in the input DataFrame
        if column in df.columns:
            # If the column is available, copy it to the result DataFrame
            result_df[column] = df[column]
        else:
            # If the column is not available, fill it with pd.NA
            result_df[column] = np.nan
    return result_df
