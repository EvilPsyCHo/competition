import pandas as pd
from tianchi.user_location_prediction.pycode.config import load_config

load_config()

def load_origin_shop():
    shop = pd.read_csv('./original/shop.csv')
    return shop

def load_original_user():
    user = pd.read_csv('./original/user_behavior.csv')
    return user

def load_original_abtest():
    abtest = pd.read_csv('./original/ab_evaluation.csv')
    return abtest


def load_sample():
    sample = pd.read_csv('./preprocessing/sample.csv')
    sample['time'] = pd.to_datetime(sample.time)
    return sample

def load_shop():
    shop = pd.read_csv('./preprocessing/shop.csv')
    return shop

