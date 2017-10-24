from tianchi.user_location_prediction.pysrc.load import *
from tianchi.user_location_prediction.pysrc.feature import *





class DataGenerator(object):

    def __init__(self):
        self.sample = load_sample()
        self.shop = load_shop()
        self.sample_wifi = load_sample_wifi_()
        self.user_cate = load_user_cate()
        self.user_price = load_user_price()

    def gen_sub_data(self, mall_id):
        sample_df = self.sample[self.sample.mall_id==mall_id].reset_index(drop=True)
        shop_df = self.shop[self.shop.mall_id==mall_id].reset_index(drop=True)
        sample_wifi_df = self.sample_wifi[self.sample_wifi.sample_id.isin(sample_df.sample_id)].reset_index(drop=True)
        user_cate_df = self.user_cate.copy()
        user_price_df = self.user_price.copy()
        return sample_df,shop_df,sample_wifi_df,user_cate_df,user_price_df

    def gen_sub_train_test_data(self, feat_name_list, mall_id):
        sample_df, shop_df, sample_wifi_df, user_cate_df, user_price_df = self.gen_sub_data(mall_id)
        feat = ext_feat(feat_name_list, sample_df=sample_df, shop_df=shop_df, sample_wifi_df=sample_wifi_df, user_price_df=user_price_df, user_cate_df=user_cate_df)
        train_idx = sample_df[sample_df.row_id.isnull()].index.tolist()[-1] + 1

        train_x = feat[:train_idx]
        train_y = sample_df[sample_df.row_id.isnull()]['shop_id'].values
        test_x = feat[train_idx:]
        test_row_id = list(sample_df[-sample_df.row_id.isnull()]['row_id'].values)
        return train_x, train_y, test_x, test_row_id





