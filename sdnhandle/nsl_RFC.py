import warnings
warnings.filterwarnings("ignore")

import numpy
from collections import defaultdict
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

#STEP 1: LOAD DATA
dataset_root = 'D://Project_Test/easyFL/benchmark/RAW_DATA/NSL-KDD'
#train_x.describe()
header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']
#------------------------创建攻击类型的映射字典-----------------------
category = defaultdict(list)
category['benign'].append('normal')


with open('training_attack_types.txt', 'r') as f:
    for line in f.readlines():
        attack, cat = line.strip().split(' ')
        category[cat].append(attack)
# print(category)
attack_mapping = dict((v,k) for k in category for v in category[k])
# print(attack_mapping)

#load train/test files
train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')
train_df = pd.read_csv(train_file, names=header_names)


#READ TRAINING DATA
import pandas as pd

train_df['label'] = train_df['attack_type'].map(lambda x: attack_mapping[x])  # 把攻击类型归到五大类中
train_df.drop(['attack_type'], axis=1, inplace=True)  # 去掉细分攻击类型那一列
train_df.drop(['success_pred'], axis=1, inplace=True)  # 去掉最后一列
# print(train_df)
# print(train_df.info())

test_df = pd.read_csv(test_file, names=header_names)
print(f'X_test shape is {test_df.shape}')
# print(Counter(test_df['attack_type']))
test_df['label'] = test_df['attack_type'].map(lambda x: attack_mapping[x])
test_df.drop(['attack_type'], axis=1, inplace=True)  # 去掉细分攻击类型那一列
test_df.drop(['success_pred'], axis=1, inplace=True)

Y_train = train_df['label'] #训练数据的标签列
Y_test  = test_df['label'] #测试数据的标签列
X_train = train_df.drop('label', axis=1) #训练数据的40列特征
X_test  = test_df.drop('label', axis=1) #测试数据的40列特征
# # ---------------------分离离散特征------------------------------
def split_category(data, columns):
    cat_data = data[columns]  # 分离出的三个离散变量
    rest_data = data.drop(columns, axis=1)  # 剩余的特征
    return rest_data, cat_data
categorical_mask = (X_train.dtypes == object)
categorical_columns = X_train.columns[categorical_mask].tolist()
# print(categorical_mask)
# print(categorical_columns)
# ----------------------把三个离散字符型特征编码，转化成数字------------------------
from sklearn.preprocessing import LabelEncoder
def label_encoder(data):
    labelencoder = LabelEncoder()
    for col in data.columns:
        data.loc[:, col] = labelencoder.fit_transform(data[col])
    return data
X_train[categorical_columns] = label_encoder(X_train[categorical_columns])
X_test[categorical_columns] = label_encoder(X_test[categorical_columns])

# ---------------------------重采样-----------------------------------
from imblearn.over_sampling import SMOTE, ADASYN
oversample = ADASYN()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)
print(f'重采样后的数据规模')
print(f'X_train shape is {X_train.shape}')
print(f'Y_train shape is {Y_train.shape}')
X_train, X_train_cat = split_category(X_train, categorical_columns)
# print(X)#剩余的38个特征
# print(X_cat) #分离出的三个离散变量  'protocol_type', 'service', 'flag'，各自的类别数为3 70 11

X_test, X_test_cat = split_category(X_test, categorical_columns)
# ----------------------------对所有离散变量进行独热编码-------------------------------
def one_hot_cat(data):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data, columns=[data.name])
    # print(data)
    out = pd.DataFrame([])
    for col in data.columns:
        one_hot_cols = pd.get_dummies(data[col], prefix=col)
        out = pd.concat([out, one_hot_cols], axis=1)
    out.set_index(data.index)
    return out

X_train_cat_one_hot = one_hot_cat(X_train_cat)
X_test_cat_one_hot = one_hot_cat(X_test_cat)


# 将测试集与训练集对齐
X_train_cat_one_hot, X_test_cat_one_hot = X_train_cat_one_hot.align(X_test_cat_one_hot, join='inner', axis=1)
X_train_cat_one_hot.fillna(0, inplace=True) #用NAN填充数据集中的空值
X_test_cat_one_hot.fillna(0, inplace=True)
X_train = pd.concat([X_train, X_train_cat_one_hot], axis=1)#数据合并
X_test = pd.concat([X_test, X_test_cat_one_hot], axis=1)
# 特征值归一化
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train) # (336803, 116)
X_test = min_max_scaler.fit_transform(X_test)

from sklearn.preprocessing import LabelEncoder
Y_train_encode = LabelEncoder().fit_transform(Y_train)
Y_test_encode = LabelEncoder().fit_transform(Y_test)


#STEP 5: CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
#--------------------把分类标签编码-------------------------------

classifier = RandomForestClassifier(n_estimators=160,
                                    max_depth=9)
classifier.fit(X_train, Y_train_encode)
pred_y = classifier.predict(X_test)

#STEP 6: Check results
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,f1_score
target_names = ['benign', 'dos', 'probe', 'r2l', 'u2r']

print(f'准确率: {accuracy_score(Y_test_encode, pred_y)}')
print(f'混淆矩阵:')
print(confusion_matrix(Y_test_encode, pred_y))
print(f'分类报告:')
print(classification_report(Y_test_encode, pred_y, target_names=target_names, digits=3))
f1 = f1_score(pred_y, Y_test_encode, average='macro')
print(f'f1_score  is {f1}')



################################ 遗传算法优化 #####################################

# coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import random
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split

generations = 10  # 繁殖代数 100
pop_size = 100  # 种群数量  500
max_value = 10  # 基因中允许出现的最大值
chrom_length = 8  # 染色体长度
pc = 0.6  # 交配概率
pm = 0.01  # 变异概率
results = [[]]  # 存储每一代的最优解，N个三元组（auc最高值, n_estimators, max_depth）
fit_value = []  # 个体适应度
fit_mean = []  # 平均适应度
pop = [[0, 1, 0, 1, 0, 1, 0, 1] for i in range(pop_size)]  # 初始化种群中所有个体的基因初始序列

# n_estimators 取 {10、20、30、40、50、60、70、80、90、100、110、120、130、140、150、160}
# max_depth 取 {1、2、3、4、5、6、7、8、9、10、11、12、13、14、15、16} 
# （1111，1111）基因组8位长

def randomForest(n_estimators_value, max_depth_value):

    # train = train.drop('label', axis=1)  # 删除训练集的类标
    # val = val.drop('label', axis=1)  # 删除测试集的类标
    global Y_test_encode

    ##  print(Y_train_encode.shape) # (336803,)

    rf = RandomForestClassifier(n_estimators=n_estimators_value,
                                max_depth=max_depth_value,
                                n_jobs=-1) # 处理器内核数量
    rf.fit(X_train, Y_train_encode)  # 训练分类器  # X_train (336803, 116)
    pred_y = rf.predict_proba(X_test)
    ## print(len(pred_y), len(Y_test_encode)) # 22544 22544

    roc_auc = metrics.roc_auc_score(Y_test_encode, pred_y, multi_class='ovr')
    return roc_auc


# Step 1 : 对参数进行编码（用于初始化基因序列，可以选择初始化基因序列，本函数省略）
def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]


# Step 2 : 计算个体的目标函数值
def cal_obj_value(pop):
    objvalue = []
    variable = decodechrom(pop)
    for i in range(len(variable)):
        tempVar = variable[i]
        n_estimators_value = (tempVar[0] + 1) * 10
        max_depth_value = tempVar[1] + 1
        aucValue = randomForest(n_estimators_value, max_depth_value)
        objvalue.append(aucValue)
    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应


# 对每个个体进行解码，并拆分成单个变量，返回 n_estimators 和 max_depth
def decodechrom(pop):
    variable = []
    n_estimators_value = []
    max_depth_value = []
    for i in range(len(pop)):
        res = []

        # 计算第一个变量值，即 0101->10(逆转)
        temp1 = pop[i][0:4]
        preValue = 0
        for pre in range(4):
            preValue += temp1[pre] * (math.pow(2, pre))
        res.append(int(preValue))

        # 计算第二个变量值
        temp2 = pop[i][4:8]
        aftValue = 0
        for aft in range(4):
            aftValue += temp2[aft] * (math.pow(2, aft))
        res.append(int(aftValue))
        variable.append(res)
    return variable


# Step 3: 计算个体的适应值（计算最大值，于是就淘汰负值就好了）
def calfitvalue(obj_value):
    fit_value = []
    temp = 0.0
    Cmin = 0
    for i in range(len(obj_value)):
        if (obj_value[i] + Cmin > 0):
            temp = Cmin + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


# Step 4: 找出适应函数值中最大值，和对应的个体
def best(pop, fit_value):
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


# Step 5: 每次繁殖，将最好的结果记录下来(将二进制转化为十进制)
def b2d(best_individual):
    temp1 = best_individual[0:4]
    preValue = 0
    for pre in range(4):
        preValue += temp1[pre] * (math.pow(2, pre))
    preValue = preValue + 1
    preValue = preValue * 10

    # 计算第二个变量值
    temp2 = best_individual[4:8]
    aftValue = 0
    for aft in range(4):
        aftValue += temp2[aft] * (math.pow(2, aft))
    aftValue = aftValue + 1
    return int(preValue), int(aftValue)


# Step 6: 自然选择（轮盘赌算法）
def selection(pop, fit_value):
    # 计算每个适应值的概率
    new_fit_value = []
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        new_fit_value.append(fit_value[i] / total_fit)
    # 计算每个适应值的累积概率
    cumsum(new_fit_value)
    # 生成随机浮点数序列
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    # 对生成的随机浮点数序列进行排序
    ms.sort()
    # 轮盘赌算法（选中的个体成为下一轮，没有被选中的直接淘汰，被选中的个体代替）
    fitin = 0
    newin = 0
    newpop = pop
    while newin < pop_len:
        if (ms[newin] < new_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop

# 求适应值的总和
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total

# 计算累积概率
def cumsum(fit_value):
    temp = []
    for i in range(len(fit_value)):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j = j + 1
        temp.append(t)
    for i in range(len(fit_value)):
        fit_value[i] = temp[i]

# Step 7: 交叉繁殖
def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0: cpoint])
            temp1.extend(pop[i + 1][cpoint: len(pop[i])])
            temp2.extend(pop[i + 1][0: cpoint])
            temp2.extend(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2

# Step 8: 基因突变
def mutation(pop, pm = 0.025):
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        mpoint = random.randint(0, py-1)
        if mpoint >= py / 2:
            pm = pm * 2
        if (random.random() < pm):
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


if __name__ == '__main__':
    # pop = geneEncoding(pop_size, chrom_length)
    for i in range(generations):
        print("第 " + str(i) + " 代开始繁殖......")
        obj_value = cal_obj_value(pop)  # 计算目标函数值
        # print(obj_value)
        fit_value = calfitvalue(obj_value)  # 计算个体的适应值
        # print(fit_value)
        [best_individual, best_fit] = best(pop, fit_value)  # 选出最好的个体和最好的函数值
        # print("best_individual: "+ str(best_individual))
        temp_n_estimator, temp_max_depth = b2d(best_individual)
        results.append([best_fit, temp_n_estimator, temp_max_depth])  # 每次繁殖，将最好的结果记录下来
        print(str(best_individual) + " " + str(best_fit))
        selection(pop, fit_value)  # 自然选择，淘汰掉一部分适应性低的个体
        crossover(pop, pc)  # 交叉繁殖
        mutation(pop, pc)  # 基因突变

    # print(results)
    results.sort()
    print(results[-1])


# 共繁衍十代，[0.9192489723801553, 110, 9]