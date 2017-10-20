import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def weekday_featext(sample_df, sample_wifi_df, shop_df):
    df = pd.DataFrame()
    df['weekday'] = sample_df.time.dt.weekday.astype(str)
    df = pd.get_dummies(df)
    feat = df.values
    return feat


def hour_featext(sample_df, sample_wifi_df, shop_df):
    df = pd.DataFrame()
    df['hour'] = sample_df.time.dt.hour.astype(str)
    df = pd.get_dummies(df)
    feat = df.values
    return feat


def spaceloc_featext(sample_df, sample_wifi_df, shop_df):
    scaler = MinMaxScaler()
    feat = scaler.fit_transform(sample_df[['lgt','ltt']])
    return feat


def spaceloc_dist_featext(sample_df, sample_wifi_df, shop_df, distfunc='squaer'):
    def squaer_dist(loc1, loc2_list):
        dist = np.power(np.sum(np.power(loc2_list-loc1,2),1),0.5)
        return dist
    user_loc = sample_df[['lgt', 'ltt']].values
    shop_loc = shop_df[['lgt', 'ltt']].values
    if distfunc=='squaer':
        feat = np.apply_along_axis(squaer_dist, 1, user_loc, **{'loc2_list':shop_loc})
    return feat


def wifi_power_featext(sample_df, sample_wifi_df, shop_df):
    feat = pd.pivot_table(sample_wifi_df, index='sample_id', columns='wifi_id', values='signal_power').fillna(0).reset_index(drop=True).values
    return feat


def wifi_powerloc_dist_featext(sample_df, sample_wifi_df, shop_df, wifi_power_feat):
    wifi_shop = pd.merge(sample_wifi_df, sample_df[['sample_id', 'shop_id']]).fillna('None')
    df = pd.pivot_table(wifi_shop, index=['shop_id', 'sample_id'], columns='wifi_id', values='signal_power').reset_index().fillna(0)
    df = df.groupby('shop_id').mean().drop('sample_id', axis=1).values
    feat = np.dot(wifi_power_feat, df.T)
    return feat


def wifi_flag_featext(sample_df, sample_wifi_df, shop_df):
    feat = pd.pivot_table(sample_wifi_df, index='sample_id', columns='wifi_id', values='signal_flag').fillna(0).reset_index(drop=True).values
    return feat


def wifi_flagloc_dist_featext(sample_df, sample_wifi_df, shop_df, wifi_flag_feat):
    wifi_shop = pd.merge(sample_wifi_df, sample_df[['sample_id', 'shop_id']]).fillna('None')
    df = pd.pivot_table(wifi_shop, index=['shop_id', 'sample_id'], columns='wifi_id', values='signal_flag').reset_index().fillna(0)
    df = df.groupby('shop_id').mean().drop('sample_id', axis=1).values
    feat = np.dot(wifi_flag_feat, df.T)
    return feat






if __name__ == '__main__':
    from tianchi.user_location_prediction.pysrc.load import *
    sample_i = load_sample_test()
    sample_wifi_i = load_sample_wifi_test()
