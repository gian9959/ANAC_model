import json
import pymysql

from anac_model.data_utils.main_utils import gen_data_indexes


def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config


def connect_to_db(db_config):
    hostname = db_config['hostname']
    username = db_config['username']
    password = db_config['password']
    database = db_config['database']

    conn = pymysql.connect(host=hostname, user=username, passwd=password, db=database)
    return conn


def all_index_files(conn):
    train_df = gen_data_indexes(conn, start_date='2016-01-01', end_date='2023-12-31')
    train_df.to_csv('csv/indexes/training_index.csv', sep=';', index=False)
    print(f'TRAINING LENGTH: {len(train_df)}')

    val_df = gen_data_indexes(conn, start_date='2024-01-01', end_date='2024-04-30')
    val_df.to_csv('csv/indexes/validation_index.csv', sep=';', index=False)
    print(f'VALIDATION LENGTH: {len(val_df)}')

    test_df = gen_data_indexes(conn, start_date='2024-05-01', end_date='2025-12-31')
    test_df.to_csv('csv/indexes/testing_index.csv', sep=';', index=False)
    print(f'TESTING LENGTH: {len(test_df)}')
