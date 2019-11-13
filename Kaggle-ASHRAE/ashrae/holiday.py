# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/10 下午2:54
"""
import holidays
import pandas as pd


locate = {
        0: {'country': 'US', 'offset': -4},
        1: {'country': 'UK', 'offset': 0},
        2: {'country': 'US', 'offset': -7},
        3: {'country': 'US', 'offset': -4},
        4: {'country': 'US', 'offset': -7},
        5: {'country': 'UK', 'offset': 0},
        6: {'country': 'US', 'offset': -4},
        7: {'country': 'CAN', 'offset': -4},
        8: {'country': 'US', 'offset': -4},
        9: {'country': 'US', 'offset': -5},
        10: {'country': 'US', 'offset': -7},
        11: {'country': 'CAN', 'offset': -4},
        12: {'country': 'IRL', 'offset': 0},
        13: {'country': 'US', 'offset': -5},
        14: {'country': 'US', 'offset': -4},
        15: {'country': 'US', 'offset': -4},
    }


def add_holiday(x):
    time_range = pd.date_range(start='2015-12-31', end='2019-01-01', freq='h')
    country_holidays = {'UK': holidays.UK(), 'US': holidays.US(), 'IRL': holidays.Ireland(), 'CAN': holidays.Canada()}

    holiday_mapping = pd.DataFrame()
    for site in range(16):
        holiday_mapping_i = pd.DataFrame({'site': site, 'timestamp': time_range})
        holiday_mapping_i['h0'] = holiday_mapping_i['timestamp'].apply(
            lambda x: x in country_holidays[locate[site]['country']]).astype(int)
        holiday_mapping = pd.concat([holiday_mapping, holiday_mapping_i], axis=0)

    x = pd.merge([x, holiday_mapping], on=['site', 'timestamp'], how='left')
    return x
