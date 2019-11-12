# 数据分析

- Site Usage

site 0 meter 0在６月前存在异常０值， site 6 meter 1 在９月前和11月后存在异常０值，site 15 在2月3月存在数据缺失。



- 时间调整

https://www.kaggle.com/patrick0302/locate-cities-according-weather-temperature

 https://www.kaggle.com/frednavruzov/aligning-temperature-timestamp 

```python
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
```

- 天气数据补全
- 数据泄露

 https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116773#latest-671028 