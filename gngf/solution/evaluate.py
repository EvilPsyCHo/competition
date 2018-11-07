# coding: utf-8
# Author: Zhirui Zhou
# Mail  : evilpsycho42@gmail.com
# Time  : 11/5/18
import numpy as np


plant_power = {
    1: 10,
    2: 10,
    3: 40,
    4: 50
}


def mae_d(df_groupby, plant):
    pm = df_groupby['pm'].values
    pp = df_groupby['pp'].values
    threshold = plant_power[plant] * 0.03
    index = pm >= threshold
    return np.abs(pm[index] - pp[index]).sum() / (np.sum(index) * plant_power[plant])


def mae_m(df, plant):
    return df.groupby(df['datetime'].dt.day).apply(lambda x: mae_d(x, plant)).mean()


def score(df, plant):
    """

    :param df: datetime, pm, pp
    :param plant:
    :return:
    """
    month = df['datetime'].dt.month.unique()
    ret = []
    for m in month:
        ret.append(mae_m(df[df['datetime'].dt.month == m], plant))
    return np.mean(ret)


def score2(pm, pp, plant):
    threshold = plant_power[plant] * 0.03
    index = pm >= threshold
    return np.abs(pm[index] - pp[index]).sum() / (np.sum(index) * plant_power[plant])
