import os
import pandas as pd
from tianchi.user_location_prediction.pycode.load_data import *

load_config()

def run_preprocessing_script():
    # load data
    shop = load_origin_shop()
    user = load_original_user()
    abtest = load_original_abtest()

    # rename columns
    shop.rename(columns={'longitude': 'lgt', 'latitude': 'ltt', 'time_stamp': 'time'}, inplace=True)
    user.rename(columns={'longitude': 'lgt', 'latitude': 'ltt', 'time_stamp': 'time'}, inplace=True)
    abtest.rename(columns={'longitude': 'lgt', 'latitude': 'ltt', 'time_stamp': 'time'}, inplace=True)
    user = user.merge(shop[['shop_id', 'mall_id']], on='shop_id', how='left')

    sample = pd.concat([user, abtest])
    sample = sample.loc[:, ['row_id', 'user_id', 'mall_id', 'time', 'lgt', 'ltt', 'wifi_infos', 'shop_id']].reset_index(drop=True)
    sample.insert(0, 'sample_id', range(sample.shape[0]))

    shop.to_csv('./preprocessing/shop.csv', index=None)
    sample.to_csv('./preprocessing/sample.csv', index=None)

if __name__ == '__main__':
    run_preprocessing_script()







