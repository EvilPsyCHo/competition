import pandas as pd


def extract_wifi_feat(df):
    sample_id_list = []
    wifi_id_list = []
    signal_power_list = []
    signal_flag_list = []

    df = df.reset_index(drop=True)
    sample_size = df.shape[0]

    for i in range(sample_size):
        wifi_info = df.wifi_infos[i].split(';')
        sample_id = df.sample_id[i]
        for w in wifi_info:
            w_values = w.split('|')
            wifi_id_list.append(w_values[0])
            signal_power_list.append(float(w_values[1]))
            signal_flag_list.append(w_values[2])
            sample_id_list.append(sample_id)

    feat = pd.DataFrame({'sample_id': sample_id_list,
                         'signal_power': signal_power_list,
                         'signal_flag': signal_flag_list,
                         'wifi_id': wifi_id_list},
                        columns=['sample_id', 'wifi_id', 'signal_power', 'signal_flag'])
    min_power = feat.signal_power.min() - 1
    max_power = feat.signal_power.max()

    feat['signal_flag'] = feat.signal_flag.apply(lambda x: 1 if x == 'true' else 0)
    feat['signal_power'] = (feat['signal_power'] - min_power) / max_power

    wifi_power = pd.pivot_table(feat, index='sample_id', columns='wifi_id', values='signal_power')
    wifi_flag = pd.pivot_table(feat, index='sample_id', columns='wifi_id', values='signal_flag')
    wifi_feat = pd.concat([wifi_power, wifi_flag], axis=1).reset_index().fillna(0)
    return wifi_feat