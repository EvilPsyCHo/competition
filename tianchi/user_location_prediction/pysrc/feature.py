import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def ext_weekday(sample_df):
    df = pd.DataFrame()
    df['weekday'] = sample_df.time.dt.weekday.astype(str)
    df = pd.get_dummies(df)
    feat = df.values
    return feat

def ext_hour(sample_df):
    df = pd.DataFrame()
    df['hour'] = sample_df.time.dt.hour.astype(str)
    df = pd.get_dummies(df)
    feat = df.values
    return feat

def ext_holiday(sample_df):
    holiday_dict = {'0':1,
                    '1':1,
                    '2':0,
                    '3':0,
                    '4':0,
                    '5':0,
                    '6':0}
    df = pd.DataFrame()
    df['holiday'] = sample_df.time.dt.weekday.astype(str).apply(lambda x: holiday_dict[x])
    feat = df.values
    return feat

def ext_space_location(sample_df):
    scaler = MinMaxScaler()
    feat = scaler.fit_transform(sample_df[['lgt','ltt']])
    return feat

def ext_space_distance(sample_df, shop_df):
    def squaer_dist(loc1, loc2_list):
        dist = np.power(np.sum(np.power(loc2_list-loc1,2),1),0.5)
        return dist
    user_loc = sample_df[['lgt', 'ltt']].values
    shop_loc = shop_df[['lgt', 'ltt']].values
    feat = np.apply_along_axis(squaer_dist, 1, user_loc, **{'loc2_list':shop_loc})
    return feat

def ext_wifi_power(sample_wifi_df):
    feat = pd.pivot_table(sample_wifi_df, index='sample_id', columns='wifi_id', values='signal_power').fillna(0).reset_index(drop=True).values
    return feat

def ext_wifi_power_dist(sample_df, sample_wifi_df):
    wifi_power_feat = ext_wifi_power(sample_wifi_df)
    wifi_shop = pd.merge(sample_wifi_df, sample_df[['sample_id', 'shop_id']]).fillna('None')
    df = pd.pivot_table(wifi_shop, index=['shop_id', 'sample_id'], columns='wifi_id', values='signal_power').reset_index().fillna(0)
    df = df.groupby('shop_id').mean().drop('sample_id', axis=1).values
    feat = np.dot(wifi_power_feat, df.T)
    return feat

def ext_wifi_flag(sample_wifi_df):
    feat = pd.pivot_table(sample_wifi_df, index='sample_id', columns='wifi_id', values='signal_flag').fillna(0).reset_index(drop=True).values
    return feat

def ext_wifi_flag_dist(sample_df, sample_wifi_df):
    wifi_flag_feat = ext_wifi_flag(sample_wifi_df)
    wifi_shop = pd.merge(sample_wifi_df, sample_df[['sample_id', 'shop_id']]).fillna('None')
    df = pd.pivot_table(wifi_shop, index=['shop_id', 'sample_id'], columns='wifi_id', values='signal_flag').reset_index().fillna(0)
    df = df.groupby('shop_id').mean().drop('sample_id', axis=1).values
    feat = np.dot(wifi_flag_feat, df.T)
    return feat

def ext_user_category(sample_df, user_cate_df):
    df = sample_df[['user_id']].merge(user_cate_df, on='user_id', how='left').fillna(0)
    feat = df.iloc[:,1:].values
    return feat

def ext_user_price(sample_df, user_price_df):
    df = sample_df[['user_id']].merge(user_price_df, on='user_id', how='left').fillna(0)
    feat = df.iloc[:,1:].values
    return feat


def ext_feat(feat_name_list, **kw):
    feat_list = []
    if 'weekday' in feat_name_list:
        weekday = ext_weekday(kw['sample_df'])
        feat_list.append(weekday)
    if 'hour' in feat_name_list:
        hour = ext_weekday(kw['sample_df'])
        feat_list.append(hour)
    if 'holiday' in feat_name_list:
        holiday = ext_holiday(kw['sample_df'])
        feat_list.append(holiday)
    if 'space_loc' in feat_name_list:
        space_loc = ext_space_location(kw['sample_df'])
        feat_list.append(space_loc)
    if 'space_dist' in feat_name_list:
        space_dist = ext_space_distance(kw['sample_df'],kw['shop_df'])
        feat_list.append(space_dist)
    if 'wifi_power' in feat_name_list:
        wifi_power = ext_wifi_power(kw['sample_wifi_df'])
        feat_list.append(wifi_power)
    if 'wifi_power_dist' in feat_name_list:
        wifi_power_dist = ext_wifi_power_dist(kw['sample_df'],kw['sample_wifi_df'])
        feat_list.append(wifi_power_dist)
    if 'wifi_flag' in feat_name_list:
        wifi_flag = ext_wifi_flag(kw['sample_wifi_df'])
        feat_list.append(wifi_flag)
    if 'wifi_falg_dist' in feat_name_list:
        wifi_falg_dist = ext_wifi_flag_dist(kw['sample_df'],kw['sample_wifi_df'])
        feat_list.append(wifi_falg_dist)
    if 'user_cate' in feat_name_list:
        user_cate = ext_user_category(kw['sample_df'], kw['user_cate_df'])
        feat_list.append(user_cate)
    if 'user_price' in feat_name_list:
        user_price = ext_user_price(kw['sample_df'], kw['user_price_df'])
        feat_list.append(user_price)

    feat = np.concatenate(feat_list, axis=1)

    return feat

