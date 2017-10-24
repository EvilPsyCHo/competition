import pandas as pd
import os

path = '/home/zhouzr/data/tianchi/user_location_predict/preprocessing_data'
os.chdir(path)

def load_sample():
    df = pd.read_csv('./sample.csv')
    df['time'] = pd.to_datetime(df.time)
    return df

def load_shop():
    df = pd.read_csv('./shop.csv')
    return df


def load_sample_wifi_():
    df = pd.read_csv('./sample_wifi.csv')
    return df

def load_shop_test():
    df = pd.read_csv('./shop_test.csv')
    return df

def load_user_cate():
    df = pd.read_csv('./user_category.csv')
    return df

def load_user_price():
    df = pd.read_csv('./user_price.csv')
    return df