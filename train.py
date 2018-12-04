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
from input_data import X_data, y_data, X_data_list, y_data_list
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# pca 降维
pca = PCA(n_components=5)
pca.fit(X_data)
X_data_pca = pca.transform(X_data)
X_data_pca_list = np.split(X_data_pca, 5)
y_data_pca_list = np.split(y_data, 5)


# check
print('X_data_pca_list len:', len(X_data_pca_list))
print('y_data_pca_list len:', len(y_data_pca_list))
for i in range(len(X_data_pca_list)):
    print('X_data_pca_list[{}] shape:'.format(i), X_data_pca_list[i].shape)
    print('y_data_pca_list[{}] shape:'.format(i), y_data_pca_list[i].shape)


def rf_func(X_data_list, y_data_list):
    """
    随机森林
    """
    print('-' * 50)
    print('rf:')
    res = []
    for n_estimators in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 500]:
        print('n_estimators = {}:'.format(n_estimators))
        clf = RandomForestClassifier(n_estimators=n_estimators)
        acc_list = []
        y_pred_all = []
        for idx, x_data in enumerate(X_data_list):
            X_test = x_data
            y_test = y_data_list[idx]
            X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
            y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
            clf.fit(X_train, y_train)
            # 验证单分类器正确率
            y_pred = clf.predict(X_test)
            y_pred_all.extend(y_pred)
            acc = (y_test == y_pred).mean() * 100
            acc_list.append(acc)
            print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
        print('mean accuracy: {:4f}'.format(np.array(acc_list).mean()))
        res.append(y_pred_all)
    return res


def knn_func(X_data_list, y_data_list):
    """
    K邻近
    """
    print('-' * 50)
    print('knn:')
    res = []
    for n_neighbors in range(1, 6):
        print('n_neighbors = {}:'.format(n_neighbors))
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        acc_list = []
        y_pred_all = []
        for idx, x_data in enumerate(X_data_list):
            X_test = x_data
            y_test = y_data_list[idx]
            X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
            y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
            clf.fit(X_train, y_train)
            # 验证单分类器正确率
            y_pred = clf.predict(X_test)
            y_pred_all.extend(y_pred)
            acc = (y_test == y_pred).mean() * 100
            acc_list.append(acc)
            print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
        print('mean accuracy: {:4f}'.format(np.array(acc_list).mean()))
        res.append(y_pred_all)
    return res


def lr_func(X_data_list, y_data_list):
    """
    逻辑回归
    """
    print('-' * 50)
    print('lr:')
    res = []
    for penalty in ['l1', 'l2']:
        acc_list = []
        y_pred_all = []
        clf = LogisticRegression(penalty=penalty, solver='liblinear')
        for idx, x_data in enumerate(X_data_list):
            X_test = x_data
            y_test = y_data_list[idx]
            X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
            y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
            clf.fit(X_train, y_train)
            # 验证单分类器正确率
            y_pred = clf.predict(X_test)
            y_pred_all.extend(y_pred)
            acc = (y_test == y_pred).mean() * 100
            acc_list.append(acc)
            print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
        print('mean accuracy: {:4f}'.format(np.array(acc_list).mean()))
        res.append(y_pred_all)
    return res


def bnb_func(X_data_list, y_data_list):
    """
    朴素贝叶斯 伯努利模型
    """
    print('-' * 50)
    print('bnb:')
    res = []
    clf = BernoulliNB()
    acc_list = []
    y_pred_all = []
    for idx, x_data in enumerate(X_data_list):
        X_test = x_data
        y_test = y_data_list[idx]
        X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
        y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
        clf.fit(X_train, y_train)
        # 验证单分类器正确率
        y_pred = clf.predict(X_test)
        y_pred_all.extend(y_pred)
        acc = (y_test == y_pred).mean() * 100
        acc_list.append(acc)
        print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
    print('mean accuracy: {:4f}'.format(np.array(acc_list).mean()))
    res.append(y_pred_all)
    return res


def mnb_func(X_data_list, y_data_list):
    """
    朴素贝叶斯 多项式模型
    """
    print('-' * 50)
    print('mnb:')
    res = []
    clf = MultinomialNB()
    acc_list = []
    y_pred_all = []
    for idx, x_data in enumerate(X_data_list):
        X_test = x_data
        y_test = y_data_list[idx]
        X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
        y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
        clf.fit(X_train, y_train)
        # 验证单分类器正确率
        y_pred = clf.predict(X_test)
        y_pred_all.extend(y_pred)
        acc = (y_test == y_pred).mean() * 100
        acc_list.append(acc)
        print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
    print('mean accuracy: {:4f}'.format(np.array(acc_list).mean()))
    res.append(y_pred_all)
    return res


def adaboost_func(X_data_list, y_data_list):
    """
    adaboost
    """
    print('-' * 50)
    print('adaboost:')
    res = []
    for n_estimators in [5, 10, 15, 20]:
        for algorithm in ['SAMME', 'SAMME.R']:
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=1), algorithm=algorithm, n_estimators=n_estimators, learning_rate=0.8)
            acc_list = []
            y_pred_all = []
            for idx, x_data in enumerate(X_data_list):
                X_test = x_data
                y_test = y_data_list[idx]
                X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
                y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
                clf.fit(X_train, y_train)
                # 验证单分类器正确率
                y_pred = clf.predict(X_test)
                y_pred_all.extend(y_pred)
                acc = (y_test == y_pred).mean() * 100
                acc_list.append(acc)
                print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
            print('mean accuracy: {:4f}'.format(np.array(acc_list).mean()))
            res.append(y_pred_all)
    return res


def light_gbm_func(X_data_list, y_data_list):
    """
    LightGBM
    """
    print('-' * 50)
    print('LightGBM:')
    res = []
    for max_depth in [3, 4, 5, 6, 7, 8, 9]:
        clf = lgb.LGBMClassifier(learning_rate=0.1, max_depth=max_depth, n_estimators=1000, objective = 'binary', reg_lambda=1)
        acc_list = []
        y_pred_all = []
        for idx, x_data in enumerate(X_data_list):
            X_test = x_data
            y_test = y_data_list[idx]
            X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
            y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
            clf.fit(X_train, y_train)
            # 验证单分类器正确率
            y_pred = clf.predict(X_test)
            y_pred_all.extend(y_pred)
            acc = (y_test == y_pred).mean() * 100
            acc_list.append(acc)
            print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
        print('mean accuracy: {:.4f}'.format(np.array(acc_list).mean()))
        res.append(y_pred_all)
    return res


def nn_func(X_data_list, y_data_list):
    """
    简单神经网络
    """
    print('-' * 50)
    print('nn:')
    input_dim = X_data_list[0].shape[1]
    res = []
    acc_list = []
    y_pred_all = []
    for idx, x_data in enumerate(X_data_list):
        model = Sequential()
        model.add(Dense(1024, input_dim=input_dim, init='uniform', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, init='uniform', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        X_test = x_data
        y_test = y_data_list[idx]
        X_train = np.vstack(X_data_list[0: idx] + X_data_list[idx + 1: ])
        y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
        model.fit(X_train, y_train, nb_epoch=150, batch_size=128, verbose=1)
        y_pred = model.predict(X_test)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = y_pred.reshape(-1,)
        y_pred_all.extend(y_pred)
        acc = (y_test == y_pred).mean() * 100
        acc_list.append(acc)
        print('cross validation {}: the accuracy is {:.4f}'.format(idx, acc))
    print('mean accuracy: {:.4f}'.format(np.array(acc_list).mean()))
    res.append(y_pred_all)
    return res


def load_res(pickle_name, pickle_files, func, X_data_list, y_data_list):
    if pickle_name in pickle_files:
        with open('./tmp/{}'.format(pickle_name), 'rb') as f:
            res = pickle.load(f)
    else:
        res = func(X_data_list, y_data_list)
        with open('./tmp/{}'.format(pickle_name), 'wb') as f:
            pickle.dump(res, f)
    return res


def stacking():

    pickle_files = os.listdir('./tmp')

    rf_res = load_res('rf.pickle', pickle_files, rf_func, X_data_list, y_data_list)
    knn_res = load_res('knn.pickle', pickle_files, knn_func, X_data_list, y_data_list)
    lr_res = load_res('lr.pickle', pickle_files, lr_func, X_data_list, y_data_list)
    bnb_res = load_res('bnb.pickle', pickle_files, bnb_func, X_data_list, y_data_list)
    mnb_res = load_res('mnb.pickle', pickle_files, mnb_func, X_data_list, y_data_list)
    # adaboost_res = load_res('adaboost.pickle', pickle_files, adaboost_func, X_data_list, y_data_list)  # 引入以后效果会变差
    lgbm_res = load_res('lgbm.pickle', pickle_files, light_gbm_func, X_data_list, y_data_list)
    nn_res = load_res('nn.pickle', pickle_files, nn_func, X_data_list, y_data_list)

    rf_pca_res = load_res('rf.pca.pickle', pickle_files, rf_func, X_data_pca_list, y_data_pca_list)
    knn_pca_res = load_res('knn.pca.pickle', pickle_files, knn_func, X_data_pca_list, y_data_pca_list)
    lr_pca_res = load_res('lr.pca.pickle', pickle_files, lr_func, X_data_pca_list, y_data_pca_list)
    # adaboost_pca_res = load_res('adaboost.pca.pickle', pickle_files, adaboost_func, X_data_pca_list, y_data_pca_list)  # 引入以后效果不提升
    lgbm_pca_res = load_res('lgbm.pca.pickle', pickle_files, light_gbm_func, X_data_pca_list, y_data_pca_list)

    predicts = []

    predicts.extend(rf_res)
    predicts.extend(knn_res)
    predicts.extend(lr_res)
    predicts.extend(lgbm_res)
    # predicts.extend(bnb_res)
    # predicts.extend(mnb_res)
    # predicts.extend(nn_res)

    predicts.extend(rf_pca_res)
    predicts.extend(knn_pca_res)
    predicts.extend(lr_pca_res)
    predicts.extend(lgbm_pca_res)

    predict_data = np.array(predicts).T
    return predict_data


def xgb_stacking():

    predict_data = stacking()
    X_data_augment = np.hstack([predict_data, X_data])
    X_data_augment_list = np.split(X_data_augment, 5)

    # check
    print('-' * 50)
    print('X_data_augment shape:', X_data_augment.shape)
    for i in range(len(X_data_augment_list)):
        print('X_data_augment_list[{}] shape:'.format(i), X_data_augment_list[i].shape)

    # 定义寻参列表
    # eta_list = [0.01, 0.1, 0.2, 0.3]
    # max_depth_list = [3, 5, 7, 9]
    # min_child_weight_list = [1, 2, 3, 4, 5]
    # gamma_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    eta_list = [0.1, 0.3]
    max_depth_list = [3]
    min_child_weight_list = [1, 2, 3, 4, 5]
    gamma_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    best_mean_accuracy = 0

    for eta in eta_list:
        for max_depth in max_depth_list:
            for min_child_weight in min_child_weight_list:
                for gamma in gamma_list:
                    # 定义参数
                    params = {
                        'objective': 'binary:logistic',
                        'eta': eta,
                        'max_depth': max_depth,
                        'min_child_weight': min_child_weight,
                        'seed': 0,
                        'class_num': 2,
                        'silent': 1,
                        'gamma': gamma
                    }
                    acc_list = []
                    for idx, x_data in enumerate(X_data_augment_list):
                        X_test = x_data
                        y_test = y_data_list[idx]
                        X_train = np.vstack(X_data_augment_list[0: idx] + X_data_augment_list[idx + 1: ])
                        y_train = np.array(y_data_list[0: idx] + y_data_list[idx + 1: ]).reshape((-1,))
                        xg_train = xgb.DMatrix(X_train, label=y_train)
                        xg_test = xgb.DMatrix(X_test, label=y_test)
                        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
                        bst = xgb.train(params, xg_train, 1000, watchlist, early_stopping_rounds=15, verbose_eval=False)
                        y_pred = bst.predict(xg_test, ntree_limit=bst.best_ntree_limit)
                        y_pred[y_pred >= 0.5] = 1
                        y_pred[y_pred < 0.5] = 0
                        acc = (y_test == y_pred).mean()
                        acc_list.append(acc)
                    mean_accuracy = np.array(acc_list).mean()
                    print('eta {}, max_depth {}, min_child_weight {}, gamma {}, acc_list {}, mean accuracy {:.5f}'.format(eta, max_depth, min_child_weight, gamma, acc_list, mean_accuracy))
                    if mean_accuracy > best_mean_accuracy:
                        best_mean_accuracy = mean_accuracy
    print('best mean accuracy: {:.5f}'.format(best_mean_accuracy))


if __name__ == '__main__':
    xgb_stacking()
