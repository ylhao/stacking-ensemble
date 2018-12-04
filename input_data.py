#encoding: utf-8
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot


FEATURE_TXT = './files/feature.txt'
FILES_PATH = './files'
DOT_MV = '.mv'
TRAIN_SAMPLE_NUM = 9000


def get_column_names():
    """
    获取列名
    return: [column_name, ...]
    """
    column_names = []
    with open(FEATURE_TXT, 'r') as f:
        for line in f:
            column_name = line.strip().split(':')[0]
            column_names.append(column_name)
    return column_names


def get_mv_dfs(column_names):
    """
    读取数据
    column_names: [column_name, ...]
    return: [DataFrame, ...]
    """
    filenames = os.listdir(FILES_PATH)
    mv_filenames = []
    for filename in filenames:
        if DOT_MV in filename:
            mv_filenames.append(filename)
    mv_dfs = []
    for filename in mv_filenames:
        df = pd.read_csv(os.path.join(FILES_PATH, filename), sep=' ', header=None, names=column_names, na_values='?', index_col=False)
        mv_dfs.append(df)
    return mv_dfs


def get_test_df(column_names):
    """
    读取数据
    column_names: [column_name, ...]
    return: [DataFrame, ...]
    """
    test_df = pd.read_csv(os.path.join(FILES_PATH, 'prelim-mv-noclass.txt'), sep=' ', header=None, names=column_names, na_values='?', index_col=False)
    return test_df


column_names = get_column_names()
mv_dfs = get_mv_dfs(column_names)
mv_df = mv_dfs[0].append(mv_dfs[1:])  # 合并数据
test_df = get_test_df(column_names)
mv_df = mv_df.append([test_df])
mv_df['class'] = mv_df['class'].fillna(-1)  # 先区分出测试集样本
print(mv_df.head(5))
print(mv_df.tail(5))
print(mv_df.shape)


conv_names = ['C{}'.format(i) for i in range(1, 5)] + ['C{}'.format(i) for i in range(31, 69)]
disc_names = [column_name for column_name in column_names if column_name not in conv_names]
disc_names.remove('class')


# check
print('-' * 50)
print('conv column num:', len(conv_names))
print('disc column num:', len(disc_names))
print('total column num:', len(conv_names) + len(disc_names))


# 拆分数据
mv_conv_df = mv_df[conv_names]
mv_disc_df = mv_df[disc_names]
labels = mv_df['class']


# check
print('-' * 50)
print('mv_conv_df shape:', mv_conv_df.shape)
print('mv_disc_df shape:', mv_disc_df.shape)
print('labels shape:', labels.shape)


# 重置索引
mv_conv_df = mv_conv_df.reset_index(drop=True)
mv_disc_df = mv_disc_df.reset_index(drop=True)


# 填补缺失值
# mv_conv_df
for column_name in mv_conv_df.columns.tolist():
    mean_val = mv_conv_df[column_name].mean()
    mv_conv_df[column_name].fillna(mean_val, inplace=True)


# mv_disc_df
# 统一填充 10000
mv_disc_df = mv_disc_df.fillna(10000)


# 数据类型转换
for column_name in mv_conv_df.columns.tolist():
    mv_conv_df[column_name] = mv_conv_df[column_name].astype(np.float64)
for column_name in mv_disc_df.columns.tolist():
    mv_disc_df[column_name] = mv_disc_df[column_name].astype(np.int32)


# check
print('-' * 50)
print(mv_conv_df.isnull().any()[mv_conv_df.isnull().any() == True])
print(mv_disc_df.isnull().any()[mv_disc_df.isnull().any() == True])


# mv_disc_df => one-hot 向量
enc = OneHotEncoder()
enc.fit(mv_disc_df)
disc_array = enc.transform(mv_disc_df).toarray()


# mv_conv_df => 归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(mv_conv_df)
conv_array = min_max_scaler.transform(mv_conv_df)


# check
print('-' * 50)
print('disc array shape:', disc_array.shape)
print('conv array shape:', conv_array.shape)


# 得到训练数据
X_temp = np.hstack((conv_array, disc_array))
y_temp = labels
X_data = X_temp[0: TRAIN_SAMPLE_NUM]
y_data = y_temp[0: TRAIN_SAMPLE_NUM]
X_data_test = X_temp[TRAIN_SAMPLE_NUM: ]


# check
print('-' * 50)
print('X_data shape:', X_data.shape)
print('y_data shape:', y_data.shape)
print('X_data_test shape:', X_data_test.shape)


# 数据集按顺序划分成 5 份（与之前的 5 份是相同的）
X_data_list = np.split(X_data, 5)
y_data_list = np.split(y_data, 5)

# check
print('X_data_list len:', len(X_data_list))
print('y_data_list len:', len(y_data_list))
for i in range(len(X_data_list)):
    print('X_data_list[{}] shape:'.format(i), X_data_list[i].shape)
    print('y_data_list[{}] shape:'.format(i), y_data_list[i].shape)

