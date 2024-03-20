import imageio
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def labeldeal(labels):
    label_list = []
    for label in labels:
        if label == 'Analysis':
            label_list.append(0)
        elif label == 'Backdoor':
            label_list.append(1)
        elif label == 'DoS':
            label_list.append(2)
        elif label == 'Exploits':
            label_list.append(3)
        elif label == 'Fuzzers':
            label_list.append(4)
        elif label == 'Generic':
            label_list.append(5)
        elif label == 'Normal':
            label_list.append(6)
        elif label == 'Reconnaissance':
            label_list.append(7)
        elif label == 'Shellcode':
            label_list.append(8)
        elif label == "Worms":
            label_list.append(9)
    return label_list

def load_data(train_set=None, test_set=None):
    # Default values.
    train_set = 'D:/Project_Test/easyFL-copy/benchmark/RAW_DATA/UNSW_NB15/UNSW_NB15_training-set.csv'
    test_set = 'D:/Project_Test/easyFL-copy/benchmark/RAW_DATA/UNSW_NB15/UNSW_NB15_testing-set.csv'
    train = pd.read_csv(train_set, index_col='id') # 指定“id”这一列数据作为行索引
    test = pd.read_csv(test_set, index_col='id') # 指定“id”这一列数据作为行索引

    print(test['attack_cat'].value_counts())
    # training_label = train['attack_cat']  # 将train的“label”这一列的值单独取出来
    # testing_label = test['attack_cat']
    # # train_label = labeldeal(training_label)
    # # test_label = labeldeal(testing_label)
    # train_label = to_categorical(train_label, 10)
    # test_label = to_categorical(test_label, 10)

    # one-hot-encoding attack label
    # Creates new dummy columns from each unique string in a particular feature 创建新的虚拟列
    unsw = pd.concat([train, test]) #将train和test拼接在一起
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])# 将'proto', 'service', 'state'这三列使用one-hot-encoder转变
    # Normalising all numerical features:

    # División del dataset por clases.
    df_class_Normal = unsw[unsw['attack_cat'] == 'Normal']
    df_class_Generic = unsw[unsw['attack_cat'] == 'Generic']
    df_class_Exploits = unsw[unsw['attack_cat'] == 'Exploits']
    df_class_Fuzzers = unsw[unsw['attack_cat'] == 'Fuzzers']
    df_class_DoS = unsw[unsw['attack_cat'] == 'DoS']
    df_class_Reconnaissance = unsw[unsw['attack_cat'] == 'Reconnaissance']
    df_class_Backdoor = unsw[unsw['attack_cat'] == 'Backdoor']
    df_class_Analysis = unsw[unsw['attack_cat'] == 'Analysis']
    df_class_Shellcode = unsw[unsw['attack_cat'] == 'Shellcode']
    df_class_Worms = unsw[unsw['attack_cat'] == 'Worms']

    # Under-sampling las categorías Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Backdoor, Analysis y Shellcode aleatoriamente a 1000 muestras
    df_class_Normal = df_class_Normal.sample(71359)

    # Concatenado
    unsw = pd.concat([df_class_Normal, df_class_Generic, df_class_Exploits, df_class_Fuzzers, df_class_DoS, df_class_Reconnaissance,
         df_class_Backdoor, df_class_Analysis, df_class_Shellcode,df_class_Worms])

    label = unsw['attack_cat']  # 将train的“label”这一列的值单独取出来
    label = labeldeal(label) # 对数据集的标签进行处理


    unsw.drop(['label', 'attack_cat'], axis=1, inplace=True) # 删除'label', 'attack_cat'这两列，其中(inplace=True)是直接对原dataFrame进行操作
    unsw_value = unsw.values
    scaler = MinMaxScaler(feature_range=(0, 1)) # 初始化MinMaxScaler
    unsw_value = scaler.fit_transform(unsw_value) # 将待处理数据矩阵进行归一化到0—1之间

    scaler = StandardScaler() # 将待处理以列为单位数据标准化，使数据服从正态分布
    unsw_value = scaler.fit_transform(unsw_value)

    train_set, test_set, train_label, test_label = train_test_split(unsw_value, label, test_size=0.2,  random_state=1)
    '''================================================================================================'''
    # train_set = unsw_value[:len(train), :] # 分离出train集
    # test_set = unsw_value[len(train):, :] # 分离出test集
    # temp_train = labels[:len(train), :].loc[197:207]
    # temp_test = labels[len(train):, :].loc[197:207]
    print(train_set.shape, len(train_label))
    print(test_set.shape, len(test_label))
    # print(train_label)
    # print(type(train_set), type(train_label))

    return train_set, train_label, test_set, test_label


load_data()
#train_set, temp_train, test_set, temp_test = load_data()
# print(train_set[1])
#
# print("----------------------------------------------------------------------------")
# train_set = train_set*255
# print(train_set[1])
# print(type(train_set), train_set.shape) # <class 'numpy.ndarray'> (82332, 196)
# # print(type(temp_train), temp_train.shape) # <class 'numpy.ndarray'> (82332,)
#
