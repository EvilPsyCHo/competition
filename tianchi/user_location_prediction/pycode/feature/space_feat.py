from sklearn.preprocessing import MinMaxScaler

def extract_space_feat(df):
    scaler = MinMaxScaler()
    feat = scaler.fit_transform(df[['lgt','ltt']])
    return feat


if __name__ == '__main__':
    from tianchi.user_location_prediction.pycode.load_data import *
    df = load_sample()
    result = extract_space_feat(df[:100])