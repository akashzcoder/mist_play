# Load the Pandas libraries with alias 'pd'
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import psycopg2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense


def _connect_to_db():
    try:
        conn = psycopg2.connect("dbname='postgres' user='test_user' host='postgres' password='test_psswd'")
    except:
        print("I am unable to connect to the database")
        raise Exception


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
    del df['os_version']

    _count_nans(df)
    fig, ax = plt.subplots()
    plt.xticks((0, 1))
    df.hist('label', ax=ax, bins=[-.5, .5, 1.5], ec="k")
    fig.savefig('example.png')
    df.to_csv(file_path)


def _create_histogram(hist):
    with open('histo.jpg', 'w') as file:
        file.write(list(hist))  # save 'hist' as a list string in a text file

def _get_predictive_set(data):
    data = shuffle(data)
    i = 8
    data_to_predict = data[:i].reset_index(drop = True)
    predict_species = data_to_predict.label
    predict_species = np.array(predict_species)
    prediction = np.array(data_to_predict.drop(['label'],axis= 1))

    data = data[i:].reset_index(drop = True)


def _count_nans(df):
    """
    only to analyse the data for the number of nan values
    :param df:
    :return: None
    """
    print(df.isnull().sum(axis=0))


def main():
    df1 = _user_table('data/user_table.csv')
    df2 = _user_app_statistics('data/user_apps_statistics.csv')
    df_interim = pd.merge(df1, df2, on="user_id", how='inner')
    print(df_interim.shape)
    df3 = _user_purchase_events('data/user_purchase_events.csv')
    df_final = pd.merge(df_interim, df3, on="user_id", how='outer')
    df_final['label'].fillna(0, inplace=True)
    df_merge = sanitizeDates(df_final)
    scaler = MinMaxScaler()
    df_merge[['date']] = scaler.fit_transform(df_merge[['date']])
    df_merge[['game_install_date']] = scaler.fit_transform(df_merge[['game_install_date']])
    _get_predictive_set(df_merge)
    _export_to_csv(file_path='data/labeled_data.csv', df=df_merge)
    trainModel(df_merge)
    #print(df_merge)

def trainModel(data):

    input = data.drop(['label'], axis = 1)
    input = np.array(input)
    result = data['label']
    train_input, test_input, train_result, test_result = model_selection.train_test_split(input,result,test_size = 0.1, random_state = 0)

    input_dim = len(data.columns) - 1
    model = Sequential()
    model.add(Dense(input_dim, input_dim = input_dim , activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['binary_accuracy'] )

    model.fit(train_input, train_result, epochs = 10, batch_size = 2)

    scores = model.evaluate(train_result, test_result)


def get_seconds(time_delta):
    return time_delta.seconds


def sanitizeDates(df_merge):
    epoch = datetime(1970, 1, 1)
    df_merge['date'] = df_merge['date'].astype(str)
    df_merge['date'] = (pd.to_datetime(df_merge['date']) - epoch).dt.total_seconds()  # .values.astype('datetime64[ms]')
    print(df_merge['date'][0])
    df_merge['date'] = df_merge['date'] - (pd.to_datetime(df_merge[
                                                              'game_install_date']) - epoch).dt.total_seconds()  # +pd.to_datetime(df_merge['game_install_timezone']).values.astype(np.int32)
    df_merge['date'] = df_merge['date'] / (3600 * 24)

    df_merge['game_install_date'] = df_merge['game_install_date'].astype(str)
    df_merge['game_install_date'] = (pd.to_datetime(
        df_merge['game_install_date']) - epoch).dt.total_seconds()  # .values.astype('datetime64[ms]')
    print(df_merge['game_install_date'][0])
    df_merge['game_install_date'] = df_merge['game_install_date'] - df_merge[
        'installed_Mistplay'] / 1000  # +pd.to_datetime(df_merge['game_install_timezone']).values.astype(np.int32)
    df_merge['game_install_date'] = df_merge['game_install_date'] / (3600 * 24)
    return df_merge


main()
