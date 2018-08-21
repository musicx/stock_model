import numpy as np
import pandas as pd
import time
# import QUANTAXIS as qa
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    train = pd.read_hdf('../data/inds/000001.hdf', key='train')

    tree = DecisionTreeClassifier(max_depth=20, min_samples_split=4, min_samples_leaf=2, random_state=33)
    train_features = [x for x in train.columns if x not in {'date', 'code', 'go_up'}]
    print(time.time())
    tree.fit(train.fillna(0).loc[:, train_features], train.loc[:, 'go_up'].fillna(0))
    print(time.time())

    test = pd.read_hdf('../data/inds/000001.hdf', key='test')
    test['pred'] = tree.predict_proba(test.loc[:, train_features])
    print(test.loc[:, ['go_up', 'pred']].value_counts())