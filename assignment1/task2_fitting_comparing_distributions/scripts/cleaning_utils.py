import pandas as pd

def clean_iris_data(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Cleans the Iris dataset by removing rows with missing values and duplicates.

    :param input_csv (str): Path to the input CSV file containing the Iris dataset.
    :type input_csv: str
    :param output_csv (str): Filename for the clean data.
    :type output_csv: str
    :returns: pd.DataFrame
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(input_csv)

    df = df.iloc[:, [2]]
    df.columns = ['sepal_width_cm']
    df.dropna(inplace=True)

    df.to_csv(f'../datasets/normal/clean/{output_csv}', index=False)

    return df
