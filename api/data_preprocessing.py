import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import psycopg2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))


def _user_table(file_path: str):
    # Read data from file 'filename.csv'
    df = pd.read_csv(file_path)
    df['bin_age'] = pd.Categorical(df['bin_age'])
    dfDummies = pd.get_dummies(df['bin_age'], prefix='age_category')
    df = pd.concat([df, dfDummies], axis=1)
    df['country_id'] = pd.Categorical(df['country_id'])
    dfDummies2 = pd.get_dummies(df['country_id'], prefix='country_id')
    df = pd.concat([df, dfDummies2], axis=1)
    del df['Unnamed: 0']  # remove the
    del df['country_id']  # remove the
    print(df.shape)
    return df


def _user_app_statistics(file_path: str):
    df = pd.read_csv(file_path)
    df['user_gross_app'] = df['n_topGrossingApps'] / df['nTotal_Apps'] + df['n_shoppingApps'] / df['nTotal_Apps']
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
    cols.insert(num_of_columns - 1, cols.pop(cols.index('label')))
    df = df.reindex(columns=cols)
    df = df[np.isfinite(df['user_gross_app'])]
    del df['bin_age']
    del df['device_id']
    del df['source_id']
    del df['game_install_timezone']
    del df['installed_Mistplay_timezone']
    del df['installed_Mistplay']

    _count_nans(df)
    fig, ax = plt.subplots()
    plt.xticks((0, 1))
    df.hist('label', ax=ax, bins=[-.5, .5, 1.5], ec="k")
    fig.savefig('example.png')
    df.to_csv(file_path)


def _create_histogram(hist):
    with open('histo.jpg', 'w') as file:
        file.write(list(hist))  # save 'hist' as a list string in a text file


def _data_balance(df):
    X = df.drop('label', axis=1)
    y = df['label']  # setting up testing and training sets
    sm = SMOTE(random_state=27, ratio=1.0)
    X, y = sm.fit_sample(X, y)
    df_final = pd.concat([X, y], axis=1)
    return df_final


def _count_nans(df):
    """
    only to analyse the data for the number of nan values
    :param df:
    :return: None
    """
    print(df.isnull().sum(axis=0))


def get_seconds(time_delta):
    return time_delta.seconds


def sanitizeDates(df_merge):
    epoch = datetime(1970, 1, 1)
    df_merge['date'] = df_merge['date'].astype(str)
    df_merge['date'] = (pd.to_datetime(df_merge['date']) - epoch).dt.total_seconds()  # .values.astype('datetime64[ms]')
    df_merge['date'] = df_merge['date'] - (pd.to_datetime(df_merge[
                                                              'game_install_date']) - epoch).dt.total_seconds()  # +pd.to_datetime(df_merge['game_install_timezone']).values.astype(np.int32)
    df_merge['date'] = df_merge['date'] / (3600 * 24)

    df_merge['game_install_date'] = df_merge['game_install_date'].astype(str)
    df_merge['game_install_date'] = (pd.to_datetime(
        df_merge['game_install_date']) - epoch).dt.total_seconds()  # .values.astype('datetime64[ms]')
    df_merge['game_install_date'] = df_merge['game_install_date'] - df_merge[
        'installed_Mistplay'] / 1000  # +pd.to_datetime(df_merge['game_install_timezone']).values.astype(np.int32)
    df_merge['game_install_date'] = df_merge['game_install_date'] / (3600 * 24)
    return df_merge


def data_preprocessing():
    user_table_dir = os.path.join(d, "data", "user_table.csv")
    user_apps_statistics_dir = os.path.join(d, "data", "user_apps_statistics.csv")
    user_purchase_events_dir = os.path.join(d, "data", "user_purchase_events.csv")
    labeled_data_dir = os.path.join(d, "data", "labeled_data.csv")
    df1 = _user_table(user_table_dir)
    df2 = _user_app_statistics(user_apps_statistics_dir)
    df_interim = pd.merge(df1, df2, on="user_id", how='inner')
    print(df_interim.shape)
    df3 = _user_purchase_events(user_purchase_events_dir)
    df_final = pd.merge(df_interim, df3, on="user_id", how='outer')
    df_final['label'].fillna(0, inplace=True)
    df_merge = sanitizeDates(df_final)
    scaler = MinMaxScaler()
    df_merge[['user_gross_app']] = scaler.fit_transform(df_merge[['user_gross_app']])
    df_merge['amount_spend'].fillna('free', inplace=True)
    del df_merge['amount_spend']
    del df_merge['user_id']
    del df_merge['os_version']
    max = df_merge['date'].max()
    df_merge['date'].fillna(max+20, inplace=True)
    df_merge['gender'].fillna(1, inplace=True)
    _export_to_csv(file_path=labeled_data_dir, df=df_merge)
    print(df_final.head())


data_preprocessing()
