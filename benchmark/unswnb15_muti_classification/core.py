from time import sleep

import numpy as np

from benchmark.unswnb15_muti_classification.data_mutil_process import load_data
from benchmark.toolkits import DefaultTaskGen, XYTaskPipe
from benchmark.toolkits import ClassificationCalculator
from benchmark.unswnb15_muti_classification.un15datesets import UN15Dataset


class TaskGen(DefaultTaskGen):
    # python generate_fedtask.py --benchmark kdd99_classification --dist 0 --skew 0 --num_clients 100
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, selected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], #  ['Analysis' 'Backdoor' 'DoS' 'Exploits' 'Fuzzers' 'Generic' 'Normal' 'Reconnaissance' 'Shellcode' 'Worms']
                 seed=0, rawdata_path='./benchmark/RAW_DATA/UNSW_NB15'):
        super(TaskGen, self).__init__(benchmark='unswnb15_muti_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path,
                                      seed=seed)
        self.num_classes = len(selected_labels)
        self.selected_labels = selected_labels
        # self.label_dict = {0: 'Analysis', 1: 'Backdoor', 2: 'DoS', 3: 'Exploits', 4: 'Fuzzers', 5: 'Generic', 6: 'Normal', 7: 'Reconnaissance', 8: 'Shellcode', 9: 'Worms',}
        self.label_dict = {0: '[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]',
                           1: '[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]',
                           2: '[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]',
                           3: '[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]',
                           4: '[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]',
                           5: '[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]',
                           6: '[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]',
                           7: '[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]',
                           8: '[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]',
                           9: '[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]' }
    def load_data(self):
        lb_convert = {}
        for i in range(len(self.selected_labels)):
            lb_convert[self.selected_labels[i]] = i

        trainx, trainy, testx, testy = load_data()  # train shape: (82332, 196) ; train_label shape:  (82332, 1)
        # trainy = np.array(trainy)
        # testy = np.array(testy)

        self.train_data = UN15Dataset(trainx, trainy)
        self.test_data = UN15Dataset(testx, testy)

        # trainy = np.array(trainy)
        # testy = np.array(testy)

        # trainy = train_label.reshape((-1, 1))
        # testy = test_label.reshape((-1, 1))
        train_didxs = [did for did in range(len(self.train_data)) if self.train_data[did][1] in self.selected_labels]
        train_data_x = [self.train_data[did][0].tolist() for did in train_didxs]
        train_data_y = [lb_convert[self.train_data[did][1]] for did in train_didxs]
        self.train_data = XYTaskPipe.TaskDataset(train_data_x, train_data_y)
        # print(type(self.train_data), self.train_data[0])  # <class 'torchvision.datasets.mnist.FashionMNIST'> <class 'tuple'> <class 'torch.Tensor'>

        test_didxs = [did for did in range(len(self.test_data)) if self.test_data[did][1] in self.selected_labels]
        test_data_x = [self.test_data[did][0].tolist() for did in test_didxs]
        test_data_y = [lb_convert[self.test_data[did][1]] for did in test_didxs]
        self.test_data = {'x': test_data_x, 'y': test_data_y}

        # self.train_data = {'x': trainx, 'y': trainy}
        # self.test_data = {'x': testx, 'y': testy}
        # print(type(self.train_data))

        # train_didxs = [did for did in range(len(self.train_data)) if self.train_data[did][1] in self.selected_labels]
        # train_data_x = [self.train_data[did][0].tolist() for did in train_didxs]
        # train_data_y = [lb_convert[self.train_data[did][1]] for did in train_didxs]
        # self.train_data = XYTaskPipe.TaskDataset(train_data_x, train_data_y)
        # # print(type(self.train_data), self.train_data[0])  # <class 'torchvision.datasets.mnist.FashionMNIST'> <class 'tuple'> <class 'torch.Tensor'>
        # test_didxs = [did for did in range(len(self.test_data)) if self.test_data[did][1] in self.selected_labels]
        # test_data_x = [self.test_data[did][0].tolist() for did in test_didxs]
        # test_data_y = [lb_convert[self.test_data[did][1]] for did in test_didxs]
        # self.test_data = {'x': test_data_x, 'y': test_data_y}

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
