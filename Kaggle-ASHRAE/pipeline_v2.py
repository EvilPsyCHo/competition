# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/12/18 10:02
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GroupKFold
import datetime as dt
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from sklearn.metrics import mean_squared_error
import gc
from pathlib import Path


# global config
DATA_PATH = Path(r"C:\Users\evilp\project\competition\Kaggle-ASHRAE\data")
# DATA_PATH = Path("../input/ashrae-energy-prediction/")

DEBUG = False
SUBMIT = True

SUMMER_START = 3000
SUMMER_END = 7500
MIN_INTERVAL = 48

ALL_FEATURE = ['month', 'hour', 'weekday', 'weekend']
USE_FEATURE = None

SEED = 42
N_SPLITS = 3

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse",
}

def log(func):
    def _wrapper(*args, **kwargs):
        print(f"start {func.__name__} ...")
        start = dt.datetime.now()
        result = func(*args, **kwargs)
        print(f"finish  {func.__name__},  cost {dt.datetime.now() - start}")
        return result
    return _wrapper

@log
def load_data(data_root_path):
    train_df = pd.read_csv(data_root_path / "train.csv", parse_dates=['timestamp'])
    test_df = pd.read_csv(data_root_path / 'test.csv', parse_dates=['timestamp'])
    weather_train = pd.read_csv(data_root_path / "weather_train.csv", parse_dates=['timestamp'])
    weather_test = pd.read_csv(data_root_path / "weather_test.csv", parse_dates=['timestamp'])
    building_meta = pd.read_csv(data_root_path / "building_metadata.csv")
    if DEBUG:
        print("use debug mode ")
        return train_df.sample(10000), test_df.sample(10000), weather_train, weather_test, building_meta
    else:
        return train_df, test_df, weather_train, weather_test, building_meta


def compress(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


@log
def process_meta(x):
    le = LabelEncoder().fit(x.primary_use)
    x['primary_use'] = le.transform(x['primary_use'])
    x['square_feet'] = np.log1p(x['square_feet'])
    x['year_built'] = 2016 - x['year_built'] + 1
    return x


@log
def process_weather(weather_df):
    weather_df = weather_df.groupby("site_id").apply(
        lambda x: x.set_index("timestamp").asfreq("H").drop("site_id", axis=1)).reset_index()

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    weather_df["hour"] = weather_df["datetime"].dt.hour

    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id', 'week', 'hour'])

    # air temperature
    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id', 'week', 'hour'])['air_temperature'].median(),
                                          columns=["air_temperature"])
    weather_df.update(air_temperature_filler, overwrite=False)

    # sea_level_pressure
    sea_level_filler = weather_df.groupby(['site_id', 'month', 'day'])['sea_level_pressure'].mean()
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])
    weather_df.update(sea_level_filler, overwrite=False)

    # wind direction
    wind_direction_filler = pd.DataFrame(weather_df.groupby(['site_id', 'month', 'day'])['wind_direction'].mean(),
                                         columns=['wind_direction'])
    weather_df.update(wind_direction_filler, overwrite=False)

    # wind speed
    wind_speed_filler = pd.DataFrame(weather_df.groupby(['site_id', 'month', 'day'])['wind_speed'].mean(),
                                     columns=['wind_speed'])
    weather_df.update(wind_speed_filler, overwrite=False)

    # precip_depth_filler
    precip_depth_filler = weather_df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].mean()
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])
    weather_df.update(precip_depth_filler, overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month', 'hour'], axis=1)

    return weather_df


def find_bad_rows(X):
    def make_is_bad_zero(x):
        meter = x.meter_id.iloc[0]

        is_zero = x.meter_reading == 0

        if meter == 0:
            return is_zero

        transitions = (is_zero != is_zero.shift(1))
        all_sequence_ids = transitions.cumsum()
        ids = all_sequence_ids[is_zero].rename("ids")

        timestamp = (x.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
        if meter in [2, 3]:
            # It's normal for steam and hotwater to be turned off during the summer
            keep = set(ids[(timestamp < SUMMER_START) |
                           (timestamp > SUMMER_END)].unique())
            is_bad = ids.isin(keep) & (ids.map(ids.value_counts()) >= MIN_INTERVAL)
        elif meter == 1:
            time_ids = ids.to_frame().join(timestamp).set_index("timestamp").ids
            is_bad = ids.map(ids.value_counts()) >= MIN_INTERVAL

            # Cold water may be turned off during the winter
            jan_id = time_ids.get(0, False)
            dec_id = time_ids.get(8283, False)
            if (jan_id and dec_id and jan_id == time_ids.get(500, False) and
                    dec_id == time_ids.get(8783, False)):
                is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))
        else:
            raise Exception(f"Unexpected meter type: {meter}")

        result = is_zero.copy()
        result.update(is_bad)
        return result

    def find_bad_zero(x):
        xx = x.assign(meter_id=x.meter)
        is_bad_zero = xx.groupby(['building_id', 'meter']).apply(make_is_bad_zero)
        idx = is_bad_zero[is_bad_zero].index.droplevel([0, 1])
        del xx
        gc.collect()
        return idx

    def find_bad_sitezero(X):
        """Returns indices of bad rows from the early days of Site 0 (UCF)."""
        timestamp = (X.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
        return X[(timestamp < 3378) & (X.site_id == 0) & (X.meter == 0)].index

    def find_bad_building1099(X):
        """Returns indices of bad rows (with absurdly high readings) from building 1099."""
        return X[(X.building_id == 1099) & (X.meter == 2) & (X.meter_reading > 3e4)].index

    return find_bad_zero(X).union(find_bad_sitezero(X)).union(find_bad_building1099(X))

@log
def process_train(x):
    assert "site_id" in x.columns
    bad_rows = find_bad_rows(x)
    x = x.drop(index=bad_rows)
    print(f'drop {len(bad_rows)} rows .')
    return x


def process_all(train_df, test_df, weather_train, weather_test, building_meta):
    weather = pd.concat([weather_train, weather_test]).reset_index(drop=True)
    weather = process_weather(weather)
    weather = compress(weather)

    building_meta = process_meta(building_meta)
    print('merge building meta')
    train_df = train_df.merge(building_meta, on='building_id', how='left')
    test_df = test_df.merge(building_meta, on='building_id', how='left')

    train_df = process_train(train_df)

    return train_df, test_df, weather, building_meta


@log
def feature_engineering(train_df, test_df, weather, building_meta):
    def add_date_feature(x):
        x["hour"] = x["timestamp"].dt.hour
        x["weekend"] = x["timestamp"].dt.weekday
        x["month"] = x['timestamp'].dt.month
        return x

    train_df = add_date_feature(train_df)
    test_df = add_date_feature(test_df)

    site_month_median = train_df.groupby(['site_id', 'month', 'meter'])['meter_reading'] \
        .median().reset_index().rename(columns={"meter_reading": "site_month_median"})
    train_df = train_df.merge(site_month_median, on=['site_id', 'month', 'meter'], how='left')
    test_df = test_df.merge(site_month_median, on=['site_id', 'month', 'meter'], how='left')

    print(f'merge weather')
    train_df = train_df.merge(weather, on=['site_id', 'timestamp'], how='left')
    test_df = test_df.merge(weather, on=['site_id', 'timestamp'], how='left')

    drop = ["timestamp", "sea_level_pressure", "wind_direction",
            "wind_speed", "year_built", "floor_count", "month"]

    train_df = train_df.drop(drop, axis=1)
    test_df = test_df.drop(drop, axis=1)
    train_target = np.log1p(train_df.meter_reading)
    row_id = test_df.row_id

    test_feature = test_df.drop('row_id', axis=1)
    train_feature = train_df.drop('meter_reading', axis=1)

    train_feature = compress(train_feature)
    test_feature = compress(test_feature)

    gc.collect()

    categorical_features = ['building_id', 'meter', 'site_id', 'primary_use', 'hour', 'weekend']
    return train_feature, train_target, test_feature, row_id, categorical_features

@log
def kfold_train(model_params, features, target, n_splits, categorical_features, seed):
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=False)
    models = []
    cv_scores = []
    for train_index, test_index in kf.split(features):
        train_features = features.loc[train_index]
        train_target = target.loc[train_index]

        test_features = features.loc[test_index]
        test_target = target.loc[test_index]

        d_training = lgb.Dataset(train_features, label=train_target, categorical_feature=categorical_features,
                                 free_raw_data=False)
        d_test = lgb.Dataset(test_features, label=test_target, categorical_feature=categorical_features,
                             free_raw_data=False)

        model = lgb.train(model_params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training, d_test],
                          verbose_eval=25, early_stopping_rounds=50)
        score = np.sqrt(mean_squared_error(test_target, model.predict(test_features, num_iteration=model.best_iteration)))
        models.append(model)
        cv_scores.append(score)
        # del train_features, train_target, test_features, test_target, d_training, d_test
        # gc.collect()

    print(f'cv score: {np.mean(cv_scores)}')
    return models, cv_scores

@log
def predict(models, features, ensemble=1):
    results = np.zeros(features.shape[0])
    for model in models:
        for i in range(ensemble):
            results += np.expm1(model.predict(features, num_iteration=model.best_iteration-i)) / (len(models) + ensemble -1)
        del model
        gc.collect()
    return results


@log
def submit(y_pred, row_id):
    assert len(y_pred) == 41697600 == len(row_id)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    submisstion = pd.DataFrame({"row_id": row_id,
                                "meter_reading": y_pred})
    submisstion.to_csv("submission.csv", index=False)


train_df, test_df, weather_train, weather_test, building_meta = load_data(DATA_PATH)
train_df, test_df, weather, building_meta = process_all(train_df, test_df, weather_train, weather_test, building_meta)
train_feature, train_target, test_feature, row_id, categorical_features = feature_engineering(train_df, test_df, weather, building_meta)
del weather
gc.collect()

models = kfold_train(params, train_feature, train_target, N_SPLITS, categorical_features, SEED)

if SUBMIT:
    preds = predict(models, test_feature)
    submit(preds, row_id)
