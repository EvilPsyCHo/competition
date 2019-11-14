# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/14 11:14
"""
import catboost as cgb
import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from tqdm import tqdm
from pathlib import Path
import random
import warnings
import logging
import time
import gc
warnings.filterwarnings('ignore')

# global config
TEST = False
SUBMISSION = False
START_TIME = time.time()
SEED = 42

# home path
# root = Path("/home/zhouzr/project/competition/Kaggle-ASHRAE/save")
# data_path = Path("/home/zhouzr/project/competition/Kaggle-ASHRAE/data/")
# raffles path
root = Path(r"C:\Users\evilp\project\competition\Kaggle-ASHRAE\save")
data_path = Path(r"C:\Users\evilp\project\competition\Kaggle-ASHRAE\data")

experiment = 'lgb_v1_cv_' + dt.strftime(dt.now(), "%m_%d_%H_%M")
experiment_path = root / experiment
experiment_path.mkdir(parents=True, exist_ok=True)
log_path = experiment_path / "log.txt"

# config logger


def get_logger():
    log = logging.Logger('lightgbm cv', level=logging.INFO)
    fmt = logging.Formatter("%(asctime)s - [line:%(lineno)d]: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    th = logging.FileHandler(filename=str(log_path), encoding='utf-8')
    th.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(th)
    return log


log = get_logger()
log.info(f'experiment: {experiment}')
log.info(f'experiment_path: {experiment_path}')
log.info(f'data_path: {data_path}')


# load data

train = pd.read_pickle(data_path / 'train.pkl')
test  = pd.read_pickle(data_path / 'test.pkl')
weather_train = pd.read_pickle(data_path / 'weather_train.pkl')
weather_test = pd.read_pickle(data_path / 'weather_test.pkl')
weather = weather_train.append(weather_test)
del weather_train, weather_test
gc.collect()

meta = pd.read_pickle(data_path / 'building_metadata.pkl')
sample_submission = pd.read_pickle(data_path / 'sample_submission.pkl')
log.info('success loading data!')

# config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


seed_everything(SEED)
use_log1p_target = True
n_folds = 2

lgb_param = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'rmse'},
        'subsample': 0.4,
        'subsample_freq': 1,
        'learning_rate': 0.25,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'seed': SEED,
            }

cgb_params = {
    'objective': 'RMSE',
    'iterations': 200,
    'learning_rate': 0.1,
    'random_seed': SEED,
    'reg_lambda': 1.0,
    'subsample': 0.5,
    'use_best_model': True,
    'depth': 6,
    'min_data_in_leaf': 1,
    'early_stopping_rounds': 50,  # int
}
categorical_features = ['h0', 'primary_use', 'hour', 'weekday', 'meter'] # , 'site_id', 'building_id'
numerical_features = ['square_feet', 'year_built', 'air_temperature', 'cloud_coverage',
              'dew_temperature', 'precip_depth_1_hr', 'floor_count']

features = numerical_features + categorical_features

log.info(f'seed: {SEED}, use_log1p_target: {use_log1p_target}, n_folds: {n_folds}')
log.info(f'features: {features}')
log.info(f'lgb params: {lgb_param}, cgb params: {cgb_params}')


# metrics
def rmsle(y_true, y_pred):
    y_true[y_true<=0] = 0
    y_pred[y_pred<0] = 0
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def lgb_rmsle(y_pred, dataset):
    y_true = dataset.label
    if use_log1p_target:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)
    y_pred[y_pred < 0] = 0.
    metric_score = rmsle(y_true, y_pred)
    is_higher_better = False
    metric_name = 'rmsle'
    return metric_name, metric_score, is_higher_better


# data processing
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


def weather_timestamp_aligned(df):
    site_offset = pd.DataFrame(locate).T.offset.to_dict()
    site_offset = df.site_id.map(site_offset)
    df['timestamp'] = df.timestamp + pd.to_timedelta(site_offset, unit='H')
    return df


weather = weather_timestamp_aligned(weather)
primary_use_encoder = LabelEncoder()
meta['primary_use'] = primary_use_encoder.fit_transform(meta['primary_use'])
if use_log1p_target:
    train['meter_reading'] = np.log1p(train['meter_reading'])
train = train.merge(meta, on='building_id', how='left')
test = test.merge(meta, on='building_id', how='left')
train = train.merge(weather, on=['site_id', 'timestamp'], how='left')
test = test.merge(weather, on=['site_id', 'timestamp'], how='left')
log.info(f'finish data processing.')


# drop nan
train = train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

# add feature


def add_holiday(x):
    time_range = pd.date_range(start='2015-12-31', end='2019-01-01', freq='h')
    country_holidays = {'UK': holidays.UK(), 'US': holidays.US(), 'IRL': holidays.Ireland(), 'CAN': holidays.Canada()}

    holiday_mapping = pd.DataFrame()
    for site in range(16):
        holiday_mapping_i = pd.DataFrame({'site_id': site, 'timestamp': time_range})
        holiday_mapping_i['h0'] = holiday_mapping_i['timestamp'].apply(
            lambda x: x in country_holidays[locate[site]['country']]).astype(int)
        holiday_mapping = pd.concat([holiday_mapping, holiday_mapping_i], axis=0)

    x = pd.merge(x, holiday_mapping, on=['site_id', 'timestamp'], how='left')
    return x


def transform(x):
    # time feature
    x['weekday'] = np.int8(x.timestamp.dt.weekday)
    x['hour'] = np.int8(x.timestamp.dt.hour)
    x['month'] = np.int8(x.timestamp.dt.month)
    return x


train = transform(train)
test = transform(test)
train = add_holiday(train)
test = add_holiday(test)
log.info(f'finish add features.')


# cgb cv
def cgb_cv(df, features, categorical_features, n_folds, param):
    kf = GroupKFold(n_splits=n_folds)
    group_map = dict(zip(np.arange(1, 13),
                         pd.cut(np.arange(1, 13), n_folds, labels=np.arange(n_folds))))
    group = df.timestamp.dt.month.map(group_map)

    models = []
    train_scores = []
    valid_scores = []

    for train_index, val_index in kf.split(df, df['building_id'], groups=group):
        train_X, train_y = df[features].iloc[train_index], df['meter_reading'].iloc[train_index]
        val_X, val_y = df[features].iloc[val_index], df['meter_reading'].iloc[val_index]

        cgb_train = cgb.Pool(train_X, train_y, cat_features=categorical_features)
        cgb_eval = cgb.Pool(val_X, val_y, cat_features=categorical_features)
        gbm = cgb.train(cgb_train, param, eval_set=cgb_eval, verbose=20)

        train_preds = gbm.predict(train_X)
        if use_log1p_target:
            train_preds = np.expm1(train_preds)
            train_y = np.expm1(train_y)
        train_scores.append(rmsle(train_y, train_preds))

        valid_preds = gbm.predict(val_X)
        if use_log1p_target:
            valid_preds = np.expm1(valid_preds)
            val_y = np.expm1(val_y)
        valid_scores.append(rmsle(val_y, valid_preds))

        models.append(gbm)
    return train_scores, valid_scores, models


if TEST:
    train = train.sample(1000, random_state=SEED)
    test = test.sample(1000, random_state=SEED)
    sample_submission = sample_submission.loc[test.index]
    print(train.head(10).index)
    print(test.head(10).index)

train_scores, valid_scores, models = cgb_cv(train, features, categorical_features, 2, cgb_params)
log.info('-' * 40 + 'cv finished!' + '-' * 40)
log.info('-' * 40 + 'cv finished!' + '-' * 40)
log.info('-' * 40 + 'cv finished!' + '-' * 40)
log.info(f'train score: {np.mean(train_scores):.3f}, valid score: {np.mean(valid_scores):.3f}')


for i, m in enumerate(models):
    model_name = f'm_{i}.txt'
    m.save_model(f'{str(experiment_path / model_name)}', num_iteration=m.best_iteration)
    lgb.plot_importance(m)
    figure_name = f'model_{i}_importance.png'
    plt.savefig(str(experiment_path / figure_name))


log.info(f'save model finished')

# submission
if SUBMISSION:
    log.info(f'predict submission ...')
    preds = np.zeros(sample_submission.shape[0]).astype('float32')
    batch = 50000
    n = int(np.ceil(sample_submission.shape[0] / batch))
    for i in tqdm(range(n)):
        start, end = i * batch, (i+1) * batch
        if use_log1p_target:
            preds[start: end] = np.mean([np.expm1(m.predict(test.iloc[start: end][features]), num_iteration=m.best_iteration)
                                         for m in models], axis=0)
        else:
            preds[start: end] = np.mean([m.predict(test.iloc[start: end][features], num_iteration=m.best_iteration)
                                         for m in models], axis=0)
    sample_submission['meter_reading'] = np.round(np.clip(preds, a_min=0, a_max=None), 1)  # clip min at zero
    sample_submission[['row_id', 'meter_reading']].to_csv(experiment_path / 'submission.csv.gz',
                                                          index=False, compression='gzip')
    print(sample_submission.shape)
    print(sample_submission.head(10))


log.info(f'Finish ALL, use time {(time.time() - START_TIME) / 60:.1f} minutes')