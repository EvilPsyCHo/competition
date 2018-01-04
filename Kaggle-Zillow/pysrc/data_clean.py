import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

def clean_flow(df):
    df = pd.concat([df, ex_nannum_feat(df)], axis=1)
    df = binary_clean(df)
    df = drop_same_feature(df)
    df = fillna_city(df)
    df = fillna_zip(df)
    return df

#data clean 前将缺省特征先提取出来
def ex_nannum_feat(df):
    t = time.time()
    print('extract nannum feat ...')
    nannum_feat = pd.DataFrame()
    for i in ['A','H','P','T','N','G','M','F','L']:
        cols = [c for c in df.columns if i in c]
        nannum_feat['*%s_nannum'%i] = df[cols].T.isnull().sum()
    nannum_feat['*total_nannum'] = nannum_feat.sum(axis=1)
    print('------  cost time %.2f seconds'%(time.time()-t))
    return nannum_feat

#二值数据处理
def binary_clean(df):
    t = time.time()
    print('binary feature clean ...')
    for c in df.columns:
        if (df[c].nunique()==1)|(c[0] == 'F'):
            df.loc[df[c].notnull(),c] = 1.
            print('convert %s'%c)
    print('------  cost time %.2f seconds'%(time.time()-t))
    return df

#高相似特征处理
def drop_same_feature(df):
    t = time.time()
    print('drop_same_feature ...')
    for f in ['A','H','P','T','N','G','M','F']:
        cols = [c for c in df.columns if c[0]==f]
        cols_len = len(cols)
        for i in range(cols_len-1):
            for j in range(1,cols_len-i):
                try:
                    corr = abs(df[[cols[i],cols[i+j]]].corr().iloc[0,1])
                    if corr > 0.95:
                        if df[cols[i]].isnull().sum()<df[cols[i+j]].isnull().sum():
                            col = cols[i]
                            col_drop = cols[i+j]
                        else:
                            col = cols[i+j]
                            col_drop = cols[i]
                        df[col] = df[col].fillna(df[col_drop])
                        df = df.drop(col_drop, axis=1)
                        print('leave feature:%s ,drop feature: %s'%(col,col_drop))
                except:
                    pass
    corr = df.corr()
    n = df.shape[1]
    corr_recorde = pd.DatsFrame(columns=['featpair','corr'])
    for i in range(1,n):
        for j in range(i-1):
            
    print('------  cost time %.2f seconds'%(time.time()-t))
    return df

def fillna_city(df):
    t = time.time()
    print('fillna city feature by knn ...')
    knn_city = KNeighborsClassifier()
    tmp = df[['G_city', 'G_latitude', 'G_longitude']].dropna().copy()
    knn_city.fit(tmp[['G_latitude', 'G_longitude']].values, tmp.G_city.values)

    fill_city = knn_city.predict(df.loc[(-df.G_latitude.isnull()) & (df.G_city.isnull()), ['G_latitude', 'G_longitude']].values)
    df.loc[(-df.G_latitude.isnull()) & (df.G_city.isnull()), 'G_city'] = fill_city
    print('------  cost time %.2f seconds'%(time.time()-t))
    return df


def fillna_zip(feat):
    t = time.time()
    print('fillna zip feature by knn ...')
    knn_zip = KNeighborsClassifier()
    tmp = feat[['G_zip', 'G_latitude', 'G_longitude']].dropna().copy()
    knn_zip.fit(tmp[['G_latitude', 'G_longitude']].values, tmp.G_zip.values)

    fill_zip = knn_zip.predict(feat.loc[(-feat.G_latitude.isnull()) & (feat.G_zip.isnull()), ['G_latitude', 'G_longitude']].values)
    feat.loc[(-feat.G_latitude.isnull()) & (feat.G_zip.isnull()), 'G_zip'] = fill_zip
    print('------  cost time %.2f seconds'%(time.time()-t))
    del knn_zip, tmp
    return feat
    
    