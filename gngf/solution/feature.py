# coding:utf8
# @Time    : 18-11-6 下午9:02
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np
from chinese_calendar import is_holiday


def arithmetic_mapping(field_1, field_2, df):
    ret = []
    features = []
    feature_cat = ['num'] * 4
    for act in '+-*/':
        ret.append(eval(f'df[field_1] {act} df[field_2]').values)
        features.append(f'{field_1}{act}{field_2}')
    ret = np.stack(ret, axis=1)
    return ret, features, feature_cat


def arithmetic_field_mapping(fields_1, fields_2, df):
    field_combination = [(f1, f2) for f1 in fields_1 for f2 in fields_2]
    ret, features, feature_cat = [], [], []
    for f1, f2 in field_combination:
        r, fs, fc = arithmetic_mapping(f1, f2, df)
        ret.append(r)
        features += fs
        feature_cat += fc
    return np.concatenate(ret, axis=1), features, feature_cat


def date_feature(df, time=True, hour=True, month=True,
         weekday=True, holiday=True, year=True):
    ret, features, feature_cat = [], [], []
    if time:
        ret.append((df.time.dt.minute + df.time.dt.hour * 60).values)
        features.append('date_time')
        feature_cat.append('num')
    if hour:
        ret.append(df.time.dt.hour)
        features.append('date_hour')
        feature_cat.append('num')
    if month:
        ret.append(df.time.dt.month)
        features.append('date_month')
        feature_cat.append('cat')
    if weekday:
        ret.append(df.time.dt.weekday)
        features.append('date_weekday')
        feature_cat.append('cat')
    if year:
        ret.append(df.time.dt.year)
        features.append('date_year')
        feature_cat.append('cat')
    if holiday:
        ret.append(df.time.apply(is_holiday).astype(np.int).values)
        features.append('date_holiday')
        feature_cat.append('cat')
    if len(ret) == 0:
        raise ValueError('必须输入至少一个特征')
    return np.stack(ret, axis=1), features, feature_cat

