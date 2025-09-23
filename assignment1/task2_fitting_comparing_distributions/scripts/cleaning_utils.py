import pandas as pd

def clean_iris_data(input_csv: str, output_csv: str):
    """
    Cleans the Iris dataset by removing rows with missing values and duplicates.

    :param input_csv: Path to the input CSV file containing the Iris dataset.
    :type input_csv: str
    :param output_csv: Filename for the clean data.
    :type output_csv: str
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

    :param input_csv: Path to the input CSV file containing the D20 dataset.
    :type input_csv: str
    :param output_csv: Filename for the clean data
    :type output_csv: str
    """
    df = pd.read_csv(input_csv)

    df.drop(['DATE', 'Day Of Week', 'ID'], inplace=True, axis=1)

    rolls = [int(x) for x in df.values.flatten() if pd.notna(x)]

    pd.DataFrame(rolls, columns=["rolls"]).to_csv(f'../datasets/uniform/clean/{output_csv}', index=False)

def clean_population_data(input_csv: str, output_csv: str):
    """
    Cleans the US city populations dataset by removing irrelevant
    features and data points.

    :param input_csv: Path to the input CSV file containing the D20 dataset.
    :type input_csv: str
    :param output_csv: Filename for the clean data
    :type output_csv: str 
    """
    df = pd.read_csv(input_csv)

    df.drop(['CITY', 'STATE', 'LAT', 'LONG'], inplace=True, axis=1)

    df.columns = ['population']
    
    df = df[df['population'] > 10]

    df.to_csv(f'../datasets/powerlaw/clean/{output_csv}', index=False)

def clean_brisbane_data(input_csv: str, output_csv: str):
    """
    Cleans/preprocesses the brisbane births dataset by transforming the minutes
    after midnight feature into a notion of intermittent arrivals of babies into
    existence.

    :param input_csv: Path to the input CSV file containing the D20 dataset.
    :type input_csv: str
    :param output_csv: Filename for the clean data
    :type output_csv: str 
    """
    df = pd.read_csv(input_csv)

    df.drop(['time_of_birth', 'gender_assigned_at_birth', 'birth_wt'], inplace=True, axis=1)
    min_after_midnight = df['min_after_midnight'].to_numpy()

    birth_intervals = [min_after_midnight[i] - min_after_midnight[i-1] for i in range (1, len(min_after_midnight))]
    birth_intervals_df = pd.DataFrame({'birth_intervals': birth_intervals})

    birth_intervals_df.to_csv(f'../datasets/exponential/clean/{output_csv}', index=False)

clean_brisbane_data('../datasets/exponential/raw/brisbane_births_intervals.csv', 'birth_intervals.csv')
