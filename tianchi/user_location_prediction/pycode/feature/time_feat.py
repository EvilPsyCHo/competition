from sklearn.preprocessing import MinMaxScaler


def extract_time_feat(df):
    df['weekday'] = df.time.dt.weekday
    df['hour'] = df.time.dt.hour
    scaler = MinMaxScaler()
    feat = scaler.fit_transform(df[['weekday','hour']])
    return feat

if __name__ == '__main__':
    pass
