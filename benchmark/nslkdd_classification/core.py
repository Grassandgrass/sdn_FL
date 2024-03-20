from time import sleep

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from benchmark.nslkdd_classification.nsl_kdd import load_data

from benchmark.toolkits import DefaultTaskGen, XYTaskPipe
from benchmark.toolkits import ClassificationCalculator


class TaskGen(DefaultTaskGen):
    # python generate_fedtask.py --benchmark kdd99_classification --dist 0 --skew 0 --num_clients 100
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, selected_labels = [0, 1], #  ['Analysis' 'Backdoor' 'DoS' 'Exploits' 'Fuzzers' 'Generic' 'Normal' 'Reconnaissance' 'Shellcode' 'Worms']
                 seed=0, rawdata_path='./benchmark/RAW_DATA/NSL-KDD'):
        super(TaskGen, self).__init__(benchmark='nslkdd_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path,
                                      seed=seed)
        self.num_classes = len(selected_labels)
        self.selected_labels = selected_labels
        # self.label_dict = {0: '0.0', 1: '1.0'}

    def load_data(self):
        lb_convert = {}
        for i in range(len(self.selected_labels)):
            lb_convert[self.selected_labels[i]] = i

        # trainx = np.load('D:/Project_Test/easyFL-copy/benchmark/nslkdd_classification/data/train-data.npy', allow_pickle=True)
        # trainy = np.load('D:/Project_Test/easyFL-copy/benchmark/nslkdd_classification/data/train-labels.npy ', allow_pickle=True)
        # testx = np.load('D:/Project_Test/easyFL-copy/benchmark/nslkdd_classification/data/test-data.npy', allow_pickle=True)
        # testy = np.load('D:/Project_Test/easyFL-copy/benchmark/nslkdd_classification/data/test-labels.npy', allow_pickle=True)

        trainx, trainy, testx, testy = load_data()


        # trainx = trainx.as_matrix()
        # trainy = trainy.as_matrix()  # <class 'numpy.ndarray'>
        # testx = testx.as_matrix()
        # testy = testy.as_matrix()  # <class 'numpy.ndarray'>

        train_didxs = [did for did in range(len(trainx)) if trainy[did] in self.selected_labels]
        train_data_x = [trainx[did].tolist() for did in train_didxs]
        train_data_y = [lb_convert[trainy[did]] for did in train_didxs]
        self.train_data = XYTaskPipe.TaskDataset(train_data_x, train_data_y)

        test_didxs = [did for did in range(len(testx)) if testy[did] in self.selected_labels]
        test_data_x = [testx[did].tolist() for did in test_didxs]
        test_data_y = [lb_convert[testy[did]] for did in test_didxs]
        # self.test_data = XYTaskPipe.TaskDataset(test_data_x, test_data_y)
        self.test_data = {'x': test_data_x, 'y': test_data_y}

        # self.train_data = {'x': trainx, 'y': trainy}
        # self.test_data = {'x': testx, 'y': testy}

    def preprocessing(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        # Shuffle the dataset
        df = df.sample(frac=1)
        # Split features and labels
        x = df.iloc[:, df.columns != 'Label']
        y = df[['Label']].to_numpy()
        # Scale the features between 0 ~ 1
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        return x, y

    def convert_data_for_saving(self):
        train_x, train_y = self.train_data.tolist()
        self.train_data = {'x': train_x, 'y': train_y}
        return

    def save_task(self, generator):  # errorFailed to saving splited dataset.
        self.convert_data_for_saving()
        XYTaskPipe.save_task(self)

class TaskPipe(XYTaskPipe):
    def __init__(self):
        super(TaskPipe, self).__init__()

class TaskCalculator(ClassificationCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
