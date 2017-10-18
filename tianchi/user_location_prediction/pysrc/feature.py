

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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
    wifi_power = pd.pivot_table(df, index='sample_id', columns='wifi_id', values='signal_power').fillna(0).reset_index(drop=True).values
    #pca = PCA(n_components=int(wifi_power.shape[1]/10))
    #wifi_power = pca.fit_transform(wifi_power)
    return wifi_power

def extract_wifi_flag(df):
    wifi_flag = pd.pivot_table(df, index='sample_id', columns='wifi_id', values='signal_flag').fillna(0).reset_index(drop=True).values
    #pca = PCA(n_components=int(wifi_flag.shape[1] / 10))
    #wifi_flag = pca.fit_transform(wifi_flag)
    return wifi_flag

if __name__ == '__main__':
    import os
    path = '/home/zhouzr/data/tianchi/user_location_predict/preprocessing_data'
    os.chdir(path)
    sample = pd.read_csv('./sample.csv')
    sample_wifi = pd.read_csv('./sample_wifi.csv')
    sample['time'] = pd.to_datetime(sample.time)
    mall = sample.mall_id.unique()[-1]
    df = sample_wifi[sample_wifi.sample_id.isin(sample[sample.mall_id == mall]['sample_id'])]
    f1 = extract_wifi_flag(df)
    f2 = extract_wifi_power(df)
    f3 = extract_time_feat(sample[sample.mall_id == mall])
    f4 = extract_space_feat(sample[sample.mall_id == mall])