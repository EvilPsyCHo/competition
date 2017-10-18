from sklearn.linear_model import LogisticRegression
import pandas as pd

from tianchi.user_location_prediction.pycode.load_data import *


sample = load_sample()

mall_list = sample.mall_id.unique()


