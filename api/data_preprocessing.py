# Load the Pandas libraries with alias 'pd'
import pandas as pd


def _user_table(file_path: str):
    # Read data from file 'filename.csv'
    df = pd.read_csv(file_path)
    df['bin_age'] = pd.Categorical(df['bin_age'])
    dfDummies = pd.get_dummies(df['bin_age'], prefix='age_category')
    df = pd.concat([df, dfDummies], axis=1)
    del df['Unnamed: 0'] # remove the
    print(df.shape)
    return df

def _user_app_statistics(file_path: str):
    df = pd.read_csv(file_path)
    df['user_gross_app'] = df['n_topGrossingApps']/df['nTotal_Apps'] + df['n_shoppingApps']/df['nTotal_Apps']
    del df['n_topGrossingApps']
    del df['nTotal_Apps']
    del df['n_shoppingApps']
    del df['Unnamed: 0']
    print(df.shape)
    return df


def _user_purchase_events(file_path: str):
    df = pd.read_csv(file_path)
    del df['Unnamed: 0']
    df.insert(0, 'label', 1)
    print(df.shape)
    return df

def _export_to_csv(file_path, df):
    cols = list(df)
    num_of_columns = len(cols)
    cols.insert(num_of_columns-1, cols.pop(cols.index('label')))
    df = df.reindex(columns=cols)
    df.to_csv(file_path)

def main():
    df1 = _user_table('/home/asingh/workspace/mist_play/mist_play/data/user_table.csv')
    df2 = _user_app_statistics('/home/asingh/workspace/mist_play/mist_play/data/user_apps_statistics.csv')
    df_interim = pd.merge(df1, df2, on="user_id", how = 'inner')
    print(df_interim.shape)
    df3 = _user_purchase_events('/home/asingh/workspace/mist_play/mist_play/data/user_purchase_events.csv')
    df_final = pd.merge(df_interim, df3, on="user_id", how = 'outer')
    df_final['label'].fillna(0, inplace=True)
    _export_to_csv(file_path='/home/asingh/workspace/mist_play/mist_play/data/labeled_data.csv', df=df_final)
    print(df_final)


main()
