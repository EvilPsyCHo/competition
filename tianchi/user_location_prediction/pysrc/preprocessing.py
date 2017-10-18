
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocessing():
    load_path = '/home/zhouzr/data/tianchi/user_location_predict/original_data'
    save_path = '/home/zhouzr/data/tianchi/user_location_predict/preprocessing_data'
    os.chdir(load_path)

    train_sample = pd.read_csv('./训练数据-ccf_first_round_user_shop_behavior.csv')
    shop = pd.read_csv('./训练数据-ccf_first_round_shop_info.csv')
    test_sample = pd.read_csv('./AB榜测试集-evaluation_public.csv')

    shop.rename(columns={'longitude': 'lgt', 'latitude': 'ltt', 'time_stamp': 'time'}, inplace=True)
    train_sample.rename(columns={'longitude': 'lgt', 'latitude': 'ltt', 'time_stamp': 'time'}, inplace=True)
    test_sample.rename(columns={'longitude': 'lgt', 'latitude': 'ltt', 'time_stamp': 'time'}, inplace=True)
    train_sample = train_sample.merge(shop[['shop_id', 'mall_id']], on='shop_id', how='left')

    sample = pd.concat([train_sample, test_sample])
    sample['sample_id'] = range(sample.shape[0])
    sample = sample.loc[:, ['sample_id', 'row_id', 'mall_id', 'user_id', 'lgt', 'ltt', 'time', 'wifi_infos', 'shop_id']]
    sample['time'] = pd.to_datetime(sample['time'])
    sample = sample.reset_index(drop=True)

    os.chdir(save_path)
    sample.to_csv('./sample.csv', index=None)
    shop.to_csv('./shop.csv', index=None)


    sample_id_list = []
    wifi_id_list = []
    signal_power_list = []
    signal_flag_list = []
    sample_size = sample.shape[0]

    for i in range(sample_size):
        wifi_info = (sample.wifi_infos[i]).split(';')
        sample_id = sample.sample_id[i]
        for w in wifi_info:
            w_values = w.split('|')
            wifi_id_list.append(w_values[0])
            signal_power_list.append(float(w_values[1]))
            signal_flag_list.append(w_values[2])
            sample_id_list.append(sample_id)

    sample_wifi = pd.DataFrame({'sample_id': sample_id_list,
                         'signal_power': signal_power_list,
                         'signal_flag': signal_flag_list,
                         'wifi_id': wifi_id_list},
                        columns=['sample_id', 'wifi_id', 'signal_power', 'signal_flag'])

    scaler_power = MinMaxScaler()
    sample_wifi['signal_power'] = scaler_power.fit_transform(sample_wifi.signal_power.values.reshape(-1,1))
    sample_wifi['signal_flag'] = sample_wifi['signal_flag'].apply(lambda x: 1 if x == 'true' else 0)
    sample_wifi.to_csv('./sample_wifi.csv', index=None)

if __name__ == '__main__':
    preprocessing()

