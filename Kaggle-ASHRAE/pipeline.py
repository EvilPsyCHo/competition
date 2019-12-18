# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/12/13 14:16
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

# DATA_PATH = "../input/ashrae-energy-prediction/"
SUBMIT = False
DEBUG = True
DATA_PATH = Path(r"C:\Users\evilp\project\competition\Kaggle-ASHRAE\data")
ALL_FEAT = []


def time_it(func):
    def _wrapper(*args, **kwargs):
        print(f"start {func.__name__} ...")
        start = dt.datetime.now()
        result = func(*args, **kwargs)
        print(f"finish  {func.__name__},  cost {dt.datetime.now() - start}")
        return result
    return _wrapper

# data
@time_it
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


def process_meta(x):
    le = LabelEncoder().fit(x.primary_use)
    x['primary_use'] = le.transform(x['primary_use'])
    x['square_feet'] = np.log1p(x['square_feet'])
    x['year_built'] = 2016 - x['year_built'] + 1
    return x


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


# processing
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

@time_it
def feature_engineering(x):
    categorical_features = ['building_id', 'meter', 'site_id', 'primary_use', 'hour', 'weekend']
    # sort values by building_id timestamp?
    x["hour"] = x["timestamp"].dt.hour
    x["weekend"] = x["timestamp"].dt.weekday
    x["month"] = x['timestamp'].dt.month

    # meter month median usage
    site_month_median = x.groupby(['site_id', 'month', 'meter'])['meter_reading']\
        .median().reset_index().rename(columns={"meter_reading": "site_month_median"})
    x = x.merge(site_month_median, on=['site_id', 'month', 'meter'], how='left')

    # meter month mean usage
    # site_month_median = x.groupby(['site_id', 'month', 'meter'])['meter_reading'] \
    #     .mean().reset_index().rename(columns={"meter_reading": "site_month_mean"})
    # x = x.merge(site_month_median, on=['site_id', 'month', 'meter'], how='left')

    drop = ["timestamp", "sea_level_pressure", "wind_direction",
            "wind_speed", "year_built", "floor_count", "month"]
    x = x.drop(drop, axis=1)
    gc.collect()
    return x, categorical_features

# train
@time_it
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


def plot_importance(models):
    for model in models:
        lgb.plot_importance(model)
        plt.show()

@time_it
def predict(models, features, ensemble=1):
    results = np.zeros(features.shape[0])
    for model in models:
        for i in range(ensemble):
            results += np.expm1(model.predict(features, num_iteration=model.best_iteration-i)) / (len(models) + ensemble -1)
        del model
        gc.collect()
    return results


def submit(pred, row_id):
    assert len(pred) == 41697600 == len(row_id)
    pred = np.clip(pred, a_min=0, a_max=None)
    submisstion = pd.DataFrame({"row_id": row_id,
                                "meter_reading": pred})
    submisstion.to_csv("submission.csv", index=False)

DATA_PATH = Path(r"C:\Users\evilp\project\competition\Kaggle-ASHRAE\data")

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse",
}

train_df, test_df, weather_train, weather_test, building_meta = load_data(DATA_PATH)

building_meta = process_meta(building_meta)
weather = pd.concat([weather_train, weather_test]).reset_index(drop=True)
del weather_test; del weather_train
gc.collect()
weather = process_weather(weather)
weather = compress(weather)

train_df = train_df[~((train_df['building_id'] == 1099) & (train_df['meter'] ==2))].reset_index(drop=True)
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")').reset_index(drop=True)
train_df = compress(train_df)
test_df = compress(test_df)

train_target = np.log1p(train_df.meter_reading)
row_id = test_df.row_id
test_df.drop('row_id', axis=1, inplace=True)

train_df = train_df.merge(building_meta, on='building_id', how='left').merge(weather, on=['site_id', 'timestamp'], how='left')
test_df = test_df.merge(building_meta, on='building_id', how='left').merge(weather, on=['site_id', 'timestamp'], how='left')

del weather
gc.collect()

feature, categorical_feature = feature_engineering(train_df)
feature.drop('meter_reading', axis=1, inplace=True)
models, score = kfold_train(params, feature, train_target, 3, categorical_feature, None)
print(f'mean cv score: {np.mean(score):.3f}')

plot_importance(models)

if SUBMIT and not DEBUG:
    print('start predict test set')
    test_feature, categorical_feature = feature_engineering(test_df)
    test_pred = predict(models, test_feature)

    print(test_pred[:10])
    submit(test_pred, row_id)
