from tianchi.user_location_prediction.pycode.feature.space_feat import *
from tianchi.user_location_prediction.pycode.feature.wifi_feat import *
from tianchi.user_location_prediction.pycode.feature.time_feat import *

def gen_train_test_data(df, mall_id):
    df_i = df[df.mall_id==mall_id].copy()
    space_feat = extract_space_feat(df_i)
    wifi_feat = extract_wifi_feat(df_i)
    time_feat = extract_time_feat(df_i)

    train_y = df_i.loc[df_i.row_id==-1, 'shop_id'].values

    train_x_df = pd.DataFrame(index=df_i.loc[df_i.row_id == -1, 'sample_id'].values)
    train_x = pd.concat([train_x_df, time_feat, space_feat, wifi_feat], axis=1).values

    test_x_df = pd.DataFrame(index=df_i[df_i.row_id != -1].sample_id)
    test_x = pd.concat([test_x_df, time_feat, space_feat, wifi_feat], axis=1).values
    row_id_list = df_i[df_i.row_id != -1].row_id.tolist()

    return train_x, train_y, test_x, row_id_list

