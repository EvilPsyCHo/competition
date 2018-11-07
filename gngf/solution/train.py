# coding:utf8
# @Time    : 18-11-6 下午6:19
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from sklearn.model_selection import KFold, ParameterGrid
import lightgbm as lgb
import numpy as np
import pandas as pd


def _get_sample_weight(y, plant, w=5):
    plant_power = {
        1: 10,
        2: 10,
        3: 40,
        4: 50
    }
    weights = np.ones_like(y)
    weights[y > plant_power[plant] * 0.03] = w
    return weights


def _score2(pm, pp, plant):
    plant_power = {
        1: 10,
        2: 10,
        3: 40,
        4: 50
    }
    threshold = plant_power[plant] * 0.03
    index = pm >= threshold
    return np.abs(pm[index] - pp[index]).sum() / (np.sum(index) * plant_power[plant])


def lgb_cv(params, x, y, plant, k=3, **kwargs):
    kf = KFold(k, **kwargs)
    weights = _get_sample_weight(y, plant)
    ret = []

    def metric(t, p):
        return _score2(t, p, plant)

    for train, valid in kf.split(x):
        train_set = lgb.Dataset(x[train], y[train], weight=weights[train], **kwargs)
        valid_set = lgb.Dataset(x[valid], y[valid], weight=weights[valid], **kwargs)
        mdl = lgb.train(params, train_set, valid_sets=[train_set, valid_set], verbose_eval=-1)
        ret.append(metric(y[valid], mdl.predict(x[valid])))
    return ret


def lgb_grid_search_cv(paras_grid, x, y, plant, k=5, **kwargs):
    grid = list(ParameterGrid(paras_grid))
    max_score = np.inf
    best_param = None
    n_step = len(grid)
    for step, p in enumerate(grid):
        score = np.mean(lgb_cv(p, x, y, plant=plant, k=k, **kwargs))
        if score < max_score:
            best_param = p
            max_score = score
            print(f'step {step / n_step * 100: .1f}%, best cv score: {max_score: .4f}')
    return best_param, max_score


def lgb_train(param, x, y, plant, **kwargs):
    weights = _get_sample_weight(y, plant)
    train_set = lgb.Dataset(x, y, weight=weights, **kwargs)
    model = lgb.train(param, train_set, verbose_eval=10)
    print(f'Plant {plant} trainset score: {_score2(y, model.predict(x), plant):.4f}')
    return model


def lgb_predict(model, x, idx):
    y = model.predict(x)
    pred = pd.DataFrame({"id": idx, "predicition": y})
    pred['id'] = pred['id'].astype(int)
    return pred
