import numpy as np
from tianchi.user_location_prediction.pysrc.feature import *


def construct_batch(sample, sample_wifi, single_mall):
    sample_single_mall = sample[sample.mall_id == single_mall].copy().reset_index(drop=True)
    sample_id_list = sample_single_mall['sample_id'].values
    sample_wifi_single_mall = sample_wifi[sample_wifi.sample_id.isin(sample_id_list)]

    time_feat = extract_time_feat(sample_single_mall)
    space_feat = extract_space_feat(sample_single_mall)
    wifi_power_feat = extract_wifi_power(sample_wifi_single_mall)
    wifi_signal_feat = extract_wifi_flag(sample_wifi_single_mall)

    feat = np.concatenate([time_feat, space_feat, wifi_power_feat, wifi_signal_feat], axis=1)

    train_idx = sample_single_mall[sample_single_mall.row_id.isnull()].index.tolist()[-1] + 1

    train_x = feat[:train_idx]
    train_y = sample_single_mall[sample_single_mall.row_id.isnull()]['shop_id'].values
    test_x = feat[train_idx:]
    test_row_id = list(sample_single_mall[-sample_single_mall.row_id.isnull()]['row_id'].values)

    return train_x, train_y, test_x, test_row_id




