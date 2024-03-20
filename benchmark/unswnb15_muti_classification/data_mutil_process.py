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
    # print(train.shape)

    # one-hot-encoding attack label
    # Creates new dummy columns from each unique string in a particular feature 创建新的虚拟列
    unsw = pd.concat([train, test]) #将train和test拼接在一起
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])# 将'proto', 'service', 'state'这三列使用one-hot-encoder转变
    print(unsw.shape)
    # Normalising all numerical features:
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
    # df_class_Analysis = df_class_Analysis.sample(1677)
    # df_class_Backdoor = df_class_Backdoor.sample(1583)
    # df_class_DoS = df_class_DoS.sample(2000)
    # df_class_Exploits = df_class_Exploits.sample(2000)
    # df_class_Fuzzers = df_class_Fuzzers.sample(2000)
    # df_class_Generic = df_class_Generic.sample(2000)
    # df_class_Reconnaissance = df_class_Reconnaissance.sample(2000)
    # df_class_Shellcode = df_class_Shellcode.sample(1378)
    # df_class_Worms = df_class_Worms.sample(174)
    # División del dataset por clases.
    # train
    df_class_Analysis_train = df_class_Analysis.iloc[:2409]
    df_class_Backdoor_train = df_class_Backdoor.iloc[:2096]
    df_class_DoS_trian = df_class_DoS.iloc[:14717]
    df_class_Exploits_train = df_class_Exploits.iloc[:20000]
    df_class_Fuzzers_train = df_class_Fuzzers.iloc[:20000]
    df_class_Generic_train = df_class_Generic.iloc[:20000]
    df_class_Normal_train = df_class_Normal.iloc[:20000]
    df_class_Reconnaissance_train = df_class_Reconnaissance.iloc[:12588]
    df_class_Shellcode_train = df_class_Shellcode.iloc[:1359]
    df_class_Worms_train = df_class_Worms.iloc[:156]
    # Concatenado
    unsw_train = pd.concat(
        [df_class_Normal_train, df_class_Generic_train, df_class_Exploits_train, df_class_Fuzzers_train, df_class_DoS_trian, df_class_Reconnaissance_train,
         df_class_Backdoor_train, df_class_Analysis_train, df_class_Shellcode_train, df_class_Worms_train])
    train_label = unsw_train['attack_cat']  # 将train的“label”这一列的值单独取出来
    print(train_label.value_counts())
    train_label = labeldeal(train_label) # 对数据集的标签进行处理
    unsw_train.drop(['label', 'attack_cat'], axis=1, inplace=True) # 删除'label', 'attack_cat'这两列，其中(inplace=True)是直接对原dataFrame进行操作
    train_unsw_value = unsw_train.values
    scaler = MinMaxScaler(feature_range=(0, 1)) # 初始化MinMaxScaler
    train_unsw_value = scaler.fit_transform(train_unsw_value) # 将待处理数据矩阵进行归一化到0—1之间
    scaler = StandardScaler() # 将待处理以列为单位数据标准化，使数据服从正态分布
    train_unsw_value = scaler.fit_transform(train_unsw_value)
    print(train_unsw_value.shape, len(train_label)) # (113325, 196) 113325
    # print(test_set.shape, len(test_label))
    # print(train_label)
    # print(type(train_set), type(train_label))


    print("==========================================================================================")
    df_class_Analysis_test = df_class_Analysis.iloc[2409:]
    df_class_Backdoor_test = df_class_Backdoor.iloc[2096:]
    df_class_DoS_test = df_class_DoS.iloc[14717:]
    df_class_Exploits_test = df_class_Exploits.iloc[20000:]
    df_class_Fuzzers_test = df_class_Fuzzers.iloc[20000:]
    df_class_Generic_test = df_class_Generic.iloc[20000:]
    df_class_Normal_test = df_class_Normal.iloc[20000:]
    df_class_Reconnaissance_test = df_class_Reconnaissance.iloc[12588:]
    df_class_Shellcode_test = df_class_Shellcode.iloc[1359:]
    df_class_Worms_test = df_class_Worms.iloc[156:]

    # Concatenado
    unsw_test = pd.concat(
        [df_class_Analysis_test, df_class_Generic_test, df_class_Exploits_test, df_class_Fuzzers_test, df_class_Normal_test,
         df_class_DoS_test, df_class_Reconnaissance_test, df_class_Backdoor_test, df_class_Shellcode_test, df_class_Worms_test])
    test_label = unsw_test['attack_cat']  # 将train的“label”这一列的值单独取出来
    print(test_label.value_counts())
    test_label = labeldeal(test_label)  # 对数据集的标签进行处理
    unsw_test.drop(['label', 'attack_cat'], axis=1, inplace=True)  # 删除'label', 'attack_cat'这两列，其中(inplace=True)是直接对原dataFrame进行操作
    test_unsw_value = unsw_test.values
    scaler = MinMaxScaler(feature_range=(0, 1))  # 初始化MinMaxScaler
    test_unsw_value = scaler.fit_transform(test_unsw_value)  # 将待处理数据矩阵进行归一化到0—1之间
    scaler = StandardScaler()  # 将待处理以列为单位数据标准化，使数据服从正态分布
    test_unsw_value = scaler.fit_transform(test_unsw_value)
    print(test_unsw_value.shape, len(test_label))  # (122707, 196) 122707
    return train_unsw_value, train_label, test_unsw_value, test_label


# load_data()
# for i in range(100):
#
#     if train_set.shape[1] == 196:
#         np.save('train_set.npy', train_set)
#         np.save('train_label.npy', train_label)
#         np.save('test_set.npy', test_set)
#         np.save('test_label.npy', test_label)
#         print(ok)
# train_set, temp_train, test_set, temp_test = load_data()
# print(train_set[1])
#
# print("----------------------------------------------------------------------------")
# train_set = train_set*255
# print(train_set[1])
# print(type(train_set), train_set.shape) # <class 'numpy.ndarray'> (82332, 196)
# # print(type(temp_train), temp_train.shape) # <class 'numpy.ndarray'> (82332,)
#
