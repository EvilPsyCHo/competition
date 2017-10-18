
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

## input sample
def extract_space_feat(df):
    scaler = MinMaxScaler()
    feat = scaler.fit_transform(df[['lgt','ltt']])
    return feat


def extract_time_feat(df):
    df['weekday'] = df.time.dt.weekday
    df['hour'] = df.time.dt.hour
    scaler = MinMaxScaler()
    feat = scaler.fit_transform(df[['weekday','hour']])
    return feat



## input sample_wifi

def extract_wifi_power(df):
    wifi_power = pd.pivot_table(df, index='sample_id', columns='wifi_id', values='signal_power').fillna(-1).drop('sample_id', axis=1).values
    return wifi_power

def extract_wifi_flag(df):
    wifi_flag = pd.pivot_table(df, index='sample_id', columns='wifi_id', values='signal_flag')