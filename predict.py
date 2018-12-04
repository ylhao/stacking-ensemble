#encoding: utf-8
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from input_data import X_temp, X_data, y_data, X_data_test
from train import stacking


# pca 降维
pca = PCA(n_components=5)
pca.fit(X_temp)
X_temp_pca = pca.transform(X_temp)


X_data_pca = X_temp_pca[0: 9000]
X_data_test_pca = X_temp_pca[9000: ]


def rf_func_test(X_train, y_train, X_test):
    """
    随机森林
    """
    print('-' * 50)
    print('rf:')
    # y_train = y_train.reshape((-1,))
    res = []
    for n_estimators in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 500]:
        print('n_estimators = {}:'.format(n_estimators))
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        res.append(y_pred)
    return res


def knn_func_test(X_train, y_train, X_test):
    """
    K邻近
    """
    print('-' * 50)
    print('knn:')
    res = []
    for n_neighbors in range(1, 6):
        print('n_neighbors = {}:'.format(n_neighbors))
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        res.append(y_pred)
    return res


def lr_func_test(X_train, y_train, X_test):
    """
    逻辑回归
    """
    print('-' * 50)
    print('lr:')
    res = []
    for penalty in ['l1', 'l2']:
        clf = LogisticRegression(penalty=penalty, solver='liblinear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        res.append(y_pred)
    return res


def light_gbm_func_test(X_train, y_train, X_test):
    """
    LightGBM
    """
    print('-' * 50)
    print('LightGBM:')
    res = []
    for max_depth in [3, 4, 5, 6, 7, 8, 9]:
        clf = lgb.LGBMClassifier(learning_rate=0.1, max_depth=max_depth, n_estimators=1000, objective = 'binary', reg_lambda=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        res.append(y_pred)
    return res


def load_res_test(pickle_name, pickle_files, func, X_data, y_data, X_data_test):
    if pickle_name in pickle_files:
        with open('./tmp/{}'.format(pickle_name), 'rb') as f:
            res = pickle.load(f)
    else:
        res = func(X_data, y_data, X_data_test)
        with open('./tmp/{}'.format(pickle_name), 'wb') as f:
            pickle.dump(res, f)
    return res


def stacking_test():

    pickle_files = os.listdir('./tmp')

    rf_res = load_res_test('rf.test.pickle', pickle_files, rf_func_test, X_data, y_data, X_data_test)
    knn_res = load_res_test('knn.test.pickle', pickle_files, knn_func_test, X_data, y_data, X_data_test)
    lr_res = load_res_test('lr.test.pickle', pickle_files, lr_func_test, X_data, y_data, X_data_test)
    lgbm_res = load_res_test('lgbm.test.pickle', pickle_files, light_gbm_func_test, X_data, y_data, X_data_test)

    rf_pca_res = load_res_test('rf.pca.test.pickle', pickle_files, rf_func_test, X_data_pca, y_data, X_data_test_pca)
    knn_pca_res = load_res_test('knn.pca.test.pickle', pickle_files, knn_func_test, X_data_pca, y_data, X_data_test_pca)
    lr_pca_res = load_res_test('lr.pca.test.pickle', pickle_files, lr_func_test, X_data_pca, y_data, X_data_test_pca)
    lgbm_pca_res = load_res_test('lgbm.pca.test.pickle', pickle_files, light_gbm_func_test, X_data_pca, y_data, X_data_test_pca)

    predicts = []

    predicts.extend(rf_res)
    predicts.extend(knn_res)
    predicts.extend(lr_res)
    predicts.extend(lgbm_res)

    predicts.extend(rf_pca_res)
    predicts.extend(knn_pca_res)
    predicts.extend(lr_pca_res)
    predicts.extend(lgbm_pca_res)

    predict_data = np.array(predicts).T
    return predict_data


def xgb_stacking():

    predict_data_train = stacking()
    predict_data_test = stacking_test()
    X_data_augment = np.hstack([predict_data_train, X_data])
    X_data_test_augment = np.hstack([predict_data_test, X_data_test])
    print(X_data_augment.shape)
    print(X_data_test_augment.shape)
    params = {
        'objective': 'binary:logistic',
        'eta': 0.3,
        'max_depth': 3,
        'min_child_weight': 3,
        'seed': 0,
        'class_num': 2,
        'silent': 1,
        'gamma': 0.1
    }
    xg_train = xgb.DMatrix(X_data_augment, label=y_data)
    xg_test = xgb.DMatrix(X_data_test_augment)
    watchlist = [(xg_train, 'train')]
    bst = xgb.train(params, xg_train, 1000, watchlist, early_stopping_rounds=15, verbose_eval=False)
    y_pred = bst.predict(xg_test)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    with open('res.txt', 'w') as f:
        for y_ in y_pred:
            f.write(str(int(y_)) + '\n')


if __name__ == '__main__':
    xgb_stacking()
