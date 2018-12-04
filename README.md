# Stacking Ensemble

这是一个 Stacking Ensemble 的实例。

## 数据的问题

数据的一个比较明显的问题就是存在大量的缺失值
数据各个特征的意义不明确
数据量不大，5 份数据合并一共 9000 条


## 尝试过的方法

1. 删掉缺失值太多的列，没有提升
2. 删掉协方差值为 0 的列，没有提升
3. 使用均值填补连续值特征的缺失值（最后采用这种方法）
4. 使用众数填补非连续值特征的缺失值（效果下降）
5. 使用中位数填补非连续值特征的缺失值（效果下降）
6. 基于每条数据的连续值特征（连续值特征已经用均值填充过缺失值）使用 KNN 算法填补非连续值特征的缺失值（效果下降）
7. 使用一个特殊的数字填补非连续值特征的缺失值（最后采用该方法）
8. 非连续值特征转为 one-hot 向量（使用该方法，有一定的提升）
9. 连续值特征归一化（使用该方法，有一定的提升）
10. 连续值特征标准化（效果不如归一化）
11. PCA 降维（使用该方法与不降维的数据分别单独使用）
13. Stacking 集成（使用该方法）
14. Stacking 中引入 adaboost 算法（效果下降）
15. Stacking 中引入 lightgbm 算法（效果上升）
16. Stacking 中引入简单神经网络（效果略有下降）
17. 扩大随机森林算法中 “n_estimators” 参数的取值范围（效果上升）
18. xgboost 算法寻参
