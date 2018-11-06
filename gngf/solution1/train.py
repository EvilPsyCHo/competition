# coding:utf8
# @Time    : 18-11-6 下午6:19
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from sklearn.model_selection import KFold, ParameterGrid
import lightgbm as lgb


def lgb_cv(params, x, y, metric, k=3, **kwargs):
    kf = KFold(k, **kwargs)
    ret = []
    for train, valid in kf.split(x):
        train_set = lgb.Dataset(x[train], y[train], **kwargs)
        valid_set = lgb.Dataset(x[valid], y[valid], **kwargs)
        mdl = lgb.train(params, train_set, valid_sets=[train_set, valid_set], verbose_eval=1)
        ret.append(metric(y[valid], mdl.predict(x[valid])))
    return ret


def lgb_grid_search_cv(paras_grid, x, y, k=3, **kwargs):
    grid = list(ParameterGrid(paras_grid))
    ret = {}
    for p in grid:
        score = lgb_cv(grid, x, y, k=k, **kwargs)
        ret[p] = score
    return ret

