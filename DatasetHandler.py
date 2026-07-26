import pandas as pd
import numpy as np


def process_dataset(csv_path):
    from pandas.api.types import is_numeric_dtype

    #read dataset
    data = pd.read_csv(csv_path)
    print(f"Before normalizing: {data.values}")

    #validate columns for numerical types
    invalid_columns = []
    for column in data.columns:
        if not is_numeric_dtype(data[column]):
            invalid_columns.append(column)

    if invalid_columns:
        return f"The following columns are not numerical: {invalid_columns}"
    else:
        processed_data = normalize_features(data)
        return processed_data

def normalize_features(data):
    for column in data.columns:
        standardized_values = []
        for value in data[column]:
            standardized_value = (value - data[column].mean())/data[column].std()
            standardized_value = (standardized_value - data[column].min())/(data[column].max() - data[column].min())
            standardized_values.append(standardized_value)
        data[column] = standardized_values
    return data