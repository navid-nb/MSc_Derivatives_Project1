import pandas as pd

def df_summary(df):
    """
    Generates summary statistics for both numerical and non-numerical columns of a DataFrame,
    similar to pandas' `describe`, but with additional details.
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to summarize.
    Returns
    -------
    num_df : pandas.DataFrame
        Summary for numerical columns, including:
            - nan_count: Number of missing (NaN) values.
            - unique_count: Number of unique values (excluding NaN).
            - zero_count: Number of zero values.
            - negative_count: Number of negative values.
            - min: Minimum value.
            - max: Maximum value.
            - mean: Mean value.
            - std: Standard deviation.
            - dtype: Data type of the column.
    non_num_df : pandas.DataFrame
        Summary for non-numerical columns, including:
            - nan_count: Number of missing (NaN) values.
            - unique_count: Number of unique values (excluding NaN).
            - empty_str_count: Number of empty string values.
            - dtype: Data type of the column.
            - most_freq: Most frequent value.
            - most_freq_count: Count of the most frequent value.
    """
    # Numerical columns
    num_cols = df.select_dtypes(include='number').columns
    num_df = pd.DataFrame({
        'nan_count': df[num_cols].isnull().sum(),
        'unique_count': df[num_cols].nunique(dropna=True),
        'zero_count': (df[num_cols] == 0).sum(),
        'negative_count': (df[num_cols] < 0).sum(),
        'min': df[num_cols].min(),
        'max': df[num_cols].max(),
        'mean': df[num_cols].mean(),
        'std': df[num_cols].std(),
        'dtype': df[num_cols].dtypes
    }).T

    # Non-numerical columns
    non_num_cols = df.select_dtypes(exclude='number').columns
    most_freq = {col: df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else None for col in non_num_cols}
    most_freq_count = {col: (df[col] == most_freq[col]).sum() if most_freq[col] is not None else 0 for col in non_num_cols}
    non_num_df = pd.DataFrame({
        'nan_count': df[non_num_cols].isnull().sum(),
        'unique_count': df[non_num_cols].nunique(dropna=True),
        'empty_str_count': (df[non_num_cols] == '').sum(),
        'dtype': df[non_num_cols].dtypes,
        'most_freq': pd.Series(most_freq),
        'most_freq_count': pd.Series(most_freq_count)
    }).T

    return num_df, non_num_df