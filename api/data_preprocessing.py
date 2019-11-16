# Load the Pandas libraries with alias 'pd'
import pandas as pd


def _user_table_processing(file_path: str):
    # Read data from file 'filename.csv'
    # (in the same directory that your python process is based)
    # Control delimiters, rows, column names with read_csv (see later)
    df = pd.read_csv(file_path)
    df['bin_age'] = pd.Categorical(df['bin_age'])
    dfDummies = pd.get_dummies(df['bin_age'], prefix='age_category')
    df = pd.concat([df, dfDummies], axis=1)
    # Preview the first 5 lines of the loaded data
    # data = data.head()
    print(df)


_user_table_processing('/home/asingh/workspace/mist_play/data/user_table.csv')
