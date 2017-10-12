import pandas as pd
import os
import time
path = '/Users/zhouzhirui/data/zillow/'
os.chdir(path)


class Load(object):
    load_featinfo_flag = False

    def load_train(self):
        t1 = time.time()
        print('load train.csv ...')
        train = pd.read_csv('train_2016_v2.csv', parse_dates=['transactiondate'])
        print('train datashape: %d X %d ,cost time: %.2fs'%(train.shape[0], train.shape[1], time.time() - t1))
        return train
    
    def load_property(self):
        if self.load_featinfo_flag:
            t1 = time.time()
            print('load properties_2016.csv ..')
            feat = pd.read_csv('properties_2016.csv')
            # rename
            name_map_dict = dict(zip(self.featinfo.Feature, self.featinfo.map_name))
            feat.columns = map(lambda x: name_map_dict[x], feat.columns)
            feat = feat.drop('G_fips', axis=1)
            print('properties datashape: %d X %d ,cost time: %.2fs'%(feat.shape[0], feat.shape[1], time.time() - t1))
            return feat
        else:
            print('请先load_featinfo')
    
    def load_submission(self):
        t1 = time.time()
        print('load sample_submission.csv ..')
        submission = pd.read_csv('sample_submission.csv')
        print('submission datashape: %d X %d ,cost time: %.2fs'%(submission.shape[0], submission.shape[1], time.time() - t1))
        return submission

    def load_featinfo(self):
        t1 = time.time()
        print('load featureInfo.csv ..')
        featinfo = pd.read_csv('featureInfo.csv')
        print('featinfo datashape: %d X %d ,cost time: %.2fs'%(featinfo.shape[0], featinfo.shape[1], time.time() - t1))
        self.load_featinfo_flag = True
        self.featinfo = featinfo
        return featinfo