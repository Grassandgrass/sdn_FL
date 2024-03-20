import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"]

# put the new label column back, 多分类
dos = ['mailbomb', 'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable',
       'snmpgetattack', 'worm']
probe = ['ipsweep', 'satan', 'nmap', 'portsweep', 'mscan', 'saint']
r2l = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock',
       'xsnoop','snmpguess', 'sendmail', 'named']
u2r = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps', 'httptunnel']

def labeldeal(labels):
    label_list = []
    for label in labels:
        if label == 'normal':
            label_list.append(0)
        elif label in dos:
            label_list.append(1)
        elif label in probe:
            label_list.append(2)
        elif label in r2l:
            label_list.append(3)
        elif label in u2r:
            label_list.append(4)
    return label_list

# def label_encoder(data):
#     labelencoder = LabelEncoder()
#     for col in data.columns:
#         data.loc[:,col] = labelencoder.fit_transform(data[col])
#     return data
# first label_encoder to allow resampling

def load_data():
    # for col_name in dataset_train.columns:
    #     if dataset_train[col_name].dtypes == 'object' :
    #         unique_cat = len(dataset_train[col_name].unique())
    #         # print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
    #
    # # Test set
    # #print('Test set:')
    # for col_name in dataset_test.columns:
    #     if dataset_test[col_name].dtypes == 'object' :
    #         unique_cat = len(dataset_test[col_name].unique())
    #         # print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
    dataset_train = pd.read_csv('D:/Project_Test/easyFL-copy/benchmark/RAW_DATA/NSL-KDD/KDDTrain+.csv', header=None,
                                names=col_names)
    dataset_test = pd.read_csv('D:/Project_Test/easyFL-copy/benchmark/RAW_DATA/NSL-KDD/KDDTest+.csv', header=None,
                               names=col_names)

    # insert code to get a list of categorical columns into a variable, categorical_columns
    categorical_columns=['protocol_type', 'service', 'flag']
     # Get the categorical values into a 2D numpy array
    dataset_train_categorical_values = dataset_train[categorical_columns]
    dataset_test_categorical_values = dataset_test[categorical_columns]

    # oversample = ADASYN()
    # dataset_train, dataset_train_categorical_values = oversample.fit_resample(dataset_train, train_y)

    # protocol type
    unique_protocol=sorted(dataset_train.protocol_type.unique())
    string1 = 'Protocol_type_'
    unique_protocol2=[string1 + x for x in unique_protocol]
    # service
    unique_service=sorted(dataset_train.service.unique())
    string2 = 'service_'
    unique_service2=[string2 + x for x in unique_service]
    # flag
    unique_flag=sorted(dataset_train.flag.unique())
    string3 = 'flag_'
    unique_flag2=[string3 + x for x in unique_flag]

    # put together
    dumcols=unique_protocol2 + unique_service2 + unique_flag2
    #do same for test set
    unique_service_test=sorted(dataset_test.service.unique())
    unique_service2_test=[string2 + x for x in unique_service_test]
    testdumcols=unique_protocol2 + unique_service2_test + unique_flag2

    #Transform categorical features into numbers using LabelEncoder()
    dataset_train_categorical_values_enc=dataset_train_categorical_values.apply(LabelEncoder().fit_transform)
    # test set
    dataset_test_categorical_values_enc=dataset_test_categorical_values.apply(LabelEncoder().fit_transform)

    #One-Hot-Encoding¶
    enc = OneHotEncoder()
    dataset_train_categorical_values_encenc = enc.fit_transform(dataset_train_categorical_values_enc)
    dataset_train_cat_data = pd.DataFrame(dataset_train_categorical_values_encenc.toarray(),columns=dumcols)
    # test set
    dataset_test_categorical_values_encenc = enc.fit_transform(dataset_test_categorical_values_enc)
    dataset_test_cat_data = pd.DataFrame(dataset_test_categorical_values_encenc.toarray(),columns=testdumcols)

    trainservice=dataset_train['service'].tolist()
    testservice= dataset_test['service'].tolist()
    difference=list(set(trainservice) - set(testservice))
    string = 'service_'
    difference=[string + x for x in difference]

    for col in difference:
        dataset_test_cat_data[col] = 0

    #Join encoded categorical dataframe with the non-categorical dataframe
    newdf=dataset_train.join(dataset_train_cat_data)
    newdf.drop('flag', axis=1, inplace=True)
    newdf.drop('protocol_type', axis=1, inplace=True)
    newdf.drop('service', axis=1, inplace=True)
    newdf.drop('num_outbound_cmds', axis=1, inplace=True)
    # newdf.drop('difficulty_level', axis=1, inplace=True)
    # test data
    newdf_test=dataset_test.join(dataset_test_cat_data)
    newdf_test.drop('flag', axis=1, inplace=True)
    newdf_test.drop('protocol_type', axis=1, inplace=True)
    newdf_test.drop('service', axis=1, inplace=True)
    newdf_test.drop('num_outbound_cmds', axis=1, inplace=True)
    # newdf_test.drop('difficulty_level', axis=1, inplace=True)

    # take label column
    train_label = newdf['label'].copy()
    # print(train_label.value_counts())
    train_label = labeldeal(train_label)

    newdf.drop('label', axis=1, inplace=True)

    test_label = newdf_test['label'].copy() # print(test_label.value_counts())
    test_label = labeldeal(test_label)
    newdf_test.drop('label', axis=1, inplace=True)


    # 归一化标准化
    scaler = MinMaxScaler(feature_range=(0, 1))  # 初始化MinMaxScaler
    train_data = scaler.fit_transform(newdf)  # 将待处理数据矩阵进行归一化到0—1之间
    test_data = scaler.fit_transform(newdf_test)  # 将待处理数据矩阵进行归一化到0—1之间

    train_data_list = []
    for trains in train_data:
        trains = np.pad(trains, 11, 'constant')
        train_data_list.append(trains)
    train_data = np.array(train_data_list)
    test_data_list = []
    for tests in test_data:
        tests = np.pad(tests, 11, 'constant')
        test_data_list.append(tests)
    test_data = np.array(test_data_list)

    oversample = ADASYN()
    # oversample = SMOTE()
    # undersample = RandomUnderSampler()
    train_data, train_label = oversample.fit_resample(train_data, train_label)
    # test_data, test_label = oversample.fit_resample(train_data, train_label)
    print(train_data.shape, len(train_label)) # (125973, 122) 125973  (336704, 122) 336704
    print(test_data.shape, len(test_label)) # (22543, 122) 22543
    np.save("data/train_data.npy", train_data)
    np.save("data/train_label.npy", train_label)
    np.save("data/test_data.npy", test_data)
    np.save("data/test_label.npy", test_label)

    return train_data, train_label, test_data, test_label

newdf, train_label, newdf_test, test_label = load_data()


# trainx = np.load('D:/Project_Test/easyFL/benchmark/nslkdd_muti_classification/data/train_data.npy', allow_pickle=True)
# trainy = np.load('D:/Project_Test/easyFL/benchmark/nslkdd_muti_classification/data/train_label.npy ', allow_pickle=True)
#
# trainy = pd.Series(trainy)
# print(trainy.value_counts())
'''
3    67355
1    67344
0    67343
4    67342
2    67320
'''