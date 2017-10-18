
import os


default_user = 'zzr_mac'

def load_config(user=default_user):
    if user == 'zzr_mac':
        os.chdir('/Users/zhouzhirui/data/tianchi/user_location_prediction')

    data_dir_list = ['original', 'preprocessing', 'feature', 'traintest', 'submission']
    for i in data_dir_list:
        if not os.path.exists(i):
            os.mkdir(i)


if __name__ == '__main__':
    load_config()