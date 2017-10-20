import numpy as np
import pandas as pd
from tianchi.user_location_prediction.pysrc.feature import *


def construct_batch(sample_df, sample_wifi_df, shop_df):
    feat_list = []

    weekday_feat = weekday_featext(sample_df, sample_wifi_df, shop_df)
    feat_list.append(weekday_feat)

    hour_feat = hour_featext(sample_df, sample_wifi_df, shop_df)
    feat_list.append(hour_feat)

    spaceloc_feat = spaceloc_featext(sample_df, sample_wifi_df, shop_df)
    feat_list.append(spaceloc_feat)

    spaceloc_dist_feat = spaceloc_dist_featext(sample_df, sample_wifi_df, shop_df)
    feat_list.append(spaceloc_dist_feat)

    wifi_power_feat = wifi_power_featext(sample_df, sample_wifi_df, shop_df)
    feat_list.append(wifi_power_feat)

    wifi_powerloc_dist_feat = wifi_powerloc_dist_featext(sample_df, sample_wifi_df, shop_df, wifi_power_feat)
    feat_list.append(wifi_powerloc_dist_feat)

    wifi_flag_feat = wifi_flag_featext(sample_df, sample_wifi_df, shop_df)
    feat_list.append(wifi_flag_feat)

    wifi_flagloc_dist_feat = wifi_flagloc_dist_featext(sample_df, sample_wifi_df, shop_df, wifi_flag_feat)
    feat_list.append(wifi_flagloc_dist_feat)

    feat = np.concatenate(feat_list, axis=1)
    train_idx = sample_df[sample_df.row_id.isnull()].index.tolist()[-1] + 1

    train_x = feat[:train_idx]
    train_y = sample_df[sample_df.row_id.isnull()]['shop_id'].values
    test_x = feat[train_idx:]
    test_row_id = list(sample_df[-sample_df.row_id.isnull()]['row_id'].values)

    return train_x, train_y, test_x, test_row_id

if __name__ == '__main__':
    from tianchi.user_location_prediction.pysrc.load import *

    sample_df = load_sample_test()
    sample_wifi_df = load_sample_wifi_test()
    shop_df = load_shop_test()




