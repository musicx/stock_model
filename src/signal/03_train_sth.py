import numpy as np
import pandas as pd
import time
# import QUANTAXIS as qa
# from sklearn.tree import DecisionTreeClassifier
import os
import xgboost as xgb

if __name__ == '__main__':

    file_names = os.listdir('../data/inds/')
    train_files = []
    test_files = []
    for name in file_names:
        if name.endswith('.hdf'):
            train_sub = pd.read_hdf('../data/inds/' + name, 'train')
            test_sub = pd.read_hdf('../data/inds/' + name, 'test')
            train_files.append(train_sub)
            test_files.append(test_sub)
    train_data = pd.concat(train_files, axis=0).fillna(0).reset_index(drop=True)
    test_data = pd.concat(test_files, axis=0).fillna(0).reset_index(drop=True)
    train_data.drop(columns=['date'], inplace=True)
    test_data.drop(columns=['date'], inplace=True)

    model = xgb.XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=50, random_state=77)
    # tree = DecisionTreeClassifier(max_depth=20, min_samples_split=4, min_samples_leaf=2, random_state=33)
    train_features = [x for x in train_data.columns if x not in {'date', 'code', 'go_up'}]
    print(time.time())
    model.fit(train_data.loc[:, train_features], train_data.loc[:, 'go_up'])
    print(time.time())
    train_data['pred'] = [x[1] for x in model.predict_proba(train_data.loc[:, train_features])]
    print(train_data.groupby(['go_up', 'pred'])['code'].count())

    test_data['pred'] = [x[1] for x in model.predict_proba(test_data.loc[:, train_features])]
    print(test_data.groupby(['go_up', 'pred'])['code'].count())
