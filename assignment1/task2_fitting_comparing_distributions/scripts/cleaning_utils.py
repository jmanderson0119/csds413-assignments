import pandas as pd

def clean_iris_data(input_csv: str, output_csv: str):
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

    df = df[df['Species'] == 'Iris-setosa']
    df = df.iloc[:, [2]]
    df.columns = ['sepal_widths_setosa']

    df.to_csv(f'../datasets/normal/clean/{output_csv}', index=False)

def clean_d20_data(input_csv: str, output_csv: str):
    """
    Cleans/preprocesses the D20 dataset by aggregating all of the roll data across
    all people into one larger random variable.
    """
    df = pd.read_csv(input_csv)

    df.drop(['DATE', 'Day Of Week', 'ID'], inplace=True, axis=1)

    rolls = [int(x) for x in df.values.flatten() if pd.notna(x)]

    pd.DataFrame(rolls, columns=["rolls"]).to_csv(f'../datasets/uniform/clean/{output_csv}', index=False)
