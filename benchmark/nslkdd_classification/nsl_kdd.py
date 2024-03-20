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

dataset_train=pd.read_csv('D:/Project_Test/easyFL-copy/benchmark/RAW_DATA/NSL-KDD/KDDTrain+.csv', header=None, names = col_names)
dataset_test=pd.read_csv('D:/Project_Test/easyFL-copy/benchmark/RAW_DATA/NSL-KDD/KDDTest+.csv', header=None, names = col_names)

# colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features
# print('Training set:')
def load_data():
    for col_name in dataset_train.columns:
        if dataset_train[col_name].dtypes == 'object' :
            unique_cat = len(dataset_train[col_name].unique())
            # print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    # Test set
    #print('Test set:')
    for col_name in dataset_test.columns:
        if dataset_test[col_name].dtypes == 'object' :
            unique_cat = len(dataset_test[col_name].unique())
            # print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    # insert code to get a list of categorical columns into a variable, categorical_columns
    categorical_columns=['protocol_type', 'service', 'flag']
     # Get the categorical values into a 2D numpy array
    dataset_train_categorical_values = dataset_train[categorical_columns]
    dataset_test_categorical_values = dataset_test[categorical_columns]

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
    labeldf=newdf['label']
    labeldf_test=newdf_test['label']
    # change the label column
    newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1,
                                 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1,
                                 'udpstorm': 1, 'worm': 1,'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,
                                 'mscan' : 2,'saint' : 2,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,
                                 'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,
                                 'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,'buffer_overflow': 4,
                                 'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
    newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                               'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                               ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                               'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})


    # put the new label column back, 二分类
    train_label = newdf['label'].copy()
    train_label[train_label != 'normal'] = 0
    train_label[train_label == 'normal'] = 1
    newdf.drop('label', axis=1, inplace=True)

    test_label = newdf_test['label'].copy()
    test_label[test_label != 'normal'] = 0
    test_label[test_label == 'normal'] = 1
    newdf_test.drop('label', axis=1, inplace=True)

    # print(newdf, newdf.shape, type(newdf)) # (125973, 121) <class 'pandas.core.frame.DataFrame'>
    # print(train_label, train_label.shape, type(train_label)) # (125973,) <class 'pandas.core.series.Series'>
    # 归一化标准化
    scaler = MinMaxScaler(feature_range=(0, 1))  # 初始化MinMaxScaler
    train_data = scaler.fit_transform(newdf)  # 将待处理数据矩阵进行归一化到0—1之间
    test_data = scaler.fit_transform(newdf_test)  # 将待处理数据矩阵进行归一化到0—1之间

    # unsw_value = scaler.fit_transform(unsw_value)
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

    return train_data, train_label, test_data, test_label

newdf, train_label, newdf_test, test_label = load_data()
print(newdf.shape)

# newdf = newdf * 255.0
# print(newdf)
# newdf['label'!= 'normal'] = 0
# newdf['label'== 'normal'] = 1
# train_label = newdf['label']
# newdf.drop('label', axis=1, inplace=True)

# newdf_test['label'!= 'normal'] = 0
# newdf_test['label'== 'normal'] = 1
# test_label = newdf_test['label']
# newdf_test.drop('label', axis=1, inplace=True)

# print(newdf, newdf.shape) # [125973 rows x 123 columns] (125973, 123)
# print(train_label, train_label.shape)


# to_drop_DoS = [2,3,4]
# to_drop_Probe = [1,3,4]
# to_drop_R2L = [1,2,4]
# to_drop_U2R = [1,2,3]
# DoS_df=newdf[~newdf['label'].isin(to_drop_DoS)];
# Probe_df=newdf[~newdf['label'].isin(to_drop_Probe)];
# R2L_df=newdf[~newdf['label'].isin(to_drop_R2L)];
# U2R_df=newdf[~newdf['label'].isin(to_drop_U2R)];
#
# #test
# DoS_df_test=newdf_test[~newdf_test['label'].isin(to_drop_DoS)];
# Probe_df_test=newdf_test[~newdf_test['label'].isin(to_drop_Probe)];
# R2L_df_test=newdf_test[~newdf_test['label'].isin(to_drop_R2L)];
# U2R_df_test=newdf_test[~newdf_test['label'].isin(to_drop_U2R)];

# print('Train:')
# print('Dimensions of DoS:' ,DoS_df.shape)
# print('Dimensions of Probe:' ,Probe_df.shape)
# print('Dimensions of R2L:' ,R2L_df.shape)
# print('Dimensions of U2R:' ,U2R_df.shape)
# print('Test:')
# print('Dimensions of DoS:' ,DoS_df_test.shape)
# print('Dimensions of Probe:' ,Probe_df_test.shape)
# print('Dimensions of R2L:' ,R2L_df_test.shape)
# print('Dimensions of U2R:' ,U2R_df_test.shape)
