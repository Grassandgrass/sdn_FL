import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time
import collections
import utils.network_simulator as ns

sample_list=['uniform', 'md']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam']
logger = None

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='mnist_classification_cnum100_dist0_skew0_seed0')
    parser.add_argument('--algorithm', help='name of algorithm;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    parser.add_argument('--pretrain', help='the path of the pretrained model parameter created by torch.save;', type=str, default='')
    # methods of server side for sampling and aggregating 服务器端的采样和聚合方法
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='md')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='uniform')
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    # hyper-parameters of training in server side 服务器端训练的超参数
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.2)
    # hyper-parameters of local training 本地训练的超参数
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--num_steps', help='the number of local steps, which dominate num_epochs when setting num_steps>0', type=int, default=-1)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=float, default='128') #128
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)

    # machine environment settings 机器学习环境设置
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', help='GPU ID, -1 for CPU', type=int, default=0)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--num_threads', help="the number of threads in the clients computing session", type=int, default=1)
    parser.add_argument('--num_workers', help='the number of workers of DataLoader', type=int, default=4)
    parser.add_argument('--test_batch_size', help='the batch_size used in testing phase;', type=int, default=512) # 512

    # the simulating system settings of clients 客户端的系统模拟设置
    # constructing the heterogeity of the network 构建网络的异质性
    parser.add_argument('--net_drop', help="controlling the dropout of clients after being selected in each communication round according to distribution Beta(drop,1)", type=float, default=0)
    parser.add_argument('--net_active', help="controlling the probability of clients being active and obey distribution Beta(active,1)", type=float, default=99999)
    parser.add_argument('--net_latency', help="controlling the variance of network conditions for different clients. The larger it is, the more differences of the network latency there are.", type=float, default=0)
    # constructing the heterogeity of computing capability
    parser.add_argument('--capability', help="controlling the difference of local computing capability of each client", type=float, default=0)

    # hyper-parameters of different algorithms 不同算法的超参数
    parser.add_argument('--learning_rate_lambda', help='η for λ in afl', type=float, default=0)
    parser.add_argument('--q', help='q in q-fedavg', type=float, default='0.0')
    parser.add_argument('--epsilon', help='ε in fedmgda+', type=float, default='0.0')
    parser.add_argument('--eta', help='global learning rate in fedmgda+', type=float, default='1.0')
    parser.add_argument('--tau', help='the length of recent history gradients to be contained in FedFAvg', type=int, default=0)
    parser.add_argument('--alpha', help='proportion of clients keeping original direction in FedFV/alpha in fedFA', type=float, default='0.0')
    parser.add_argument('--beta', help='beta in FedFA',type=float, default='1.0')
    parser.add_argument('--gamma', help='gamma in FedFA', type=float, default='0')
    parser.add_argument('--mu', help='mu in fedprox', type=float, default='0.1')
    parser.add_argument('--alg', help='clustered sampling', type=int, default=1)
    parser.add_argument('--w', help='whether to wait for all updates being initialized before aggregation', type=int, default=1)
    parser.add_argument('--c', help='proportion of clients keeping original direction in FedFV/alpha in fedFA', type=float, default='0.0')
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)

def initialize(option):
    # init fedtask
    print("init fedtask...", end='')
    # dynamical initializing the configuration with the benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:{}'.format(option['gpu']) if torch.cuda.is_available() and option['gpu'] != -1 else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
    # The Model is defined in bmk_model_path as default, whose filename is option['model'] and the classname is 'Model'
    # If an algorithm change the backbone for a task, a modified model should be defined in the path 'algorithm/method_name.py', whose classname is option['model']
    try:
        utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    except ModuleNotFoundError:
        utils.fmodule.Model = getattr(importlib.import_module('.'.join(['algorithm', option['algorithm']])), option['model'])
    model = utils.fmodule.Model().to(utils.fmodule.device)
    # init the model that owned by the server (e.g. the model trained in the server-side在服务端训练的模型)
    try:
        utils.fmodule.SvrModel = getattr(importlib.import_module(bmk_model_path), 'SvrModel')
    except:
        utils.fmodule.SvrModel = utils.fmodule.Model
    # init the model that owned by the client (e.g. the personalized model whose type may be different from the global model类型可能不同于全局模型的个性化模型)
    try:
        utils.fmodule.CltModel = getattr(importlib.import_module(bmk_model_path), 'CltModel')
    except:
        utils.fmodule.CltModel = utils.fmodule.Model
    # load pre-trained model
    try:
        if option['pretrain'] != '':
            model.load_state_dict(torch.load(option['pretrain'])['model'])
    except:
        print("Invalid Model Configuration.")
        exit(1)
    # read federated task by TaskPipe
    TaskPipe = getattr(importlib.import_module(bmk_core_path), 'TaskPipe')
    train_datas, valid_datas, test_data, client_names = TaskPipe.load_task(os.path.join('fedtask', option['task']))
    num_clients = len(client_names)
    print("done")

    # init client
    print('init clients...', end='')
    client_path = '%s.%s' % ('algorithm', option['algorithm'])
    Client=getattr(importlib.import_module(client_path), 'Client')
    clients = [Client(option, name = client_names[cid], train_data = train_datas[cid], valid_data = valid_datas[cid]) for cid in range(num_clients)]
    print('done')

    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, model, clients, test_data = test_data)
    # init virtual network environment
    ns.init_network_environment(server)
    # init logger
    try:
        Logger = getattr(importlib.import_module(server_path), 'MyLogger')
    except AttributeError:
        Logger = DefaultLogger
    global logger
    logger = Logger()
    print('done')
    return server

def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para in server.paras_name: header = header + para + "{}_".format(option[para])
    output_name = header + "M{}_R{}_B{}_E{}_NS{}_LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_DR{:.2f}_AC{:.2f}_.json".format(
        option['model'],
        option['num_rounds'],
        option['batch_size'],
        option['num_epochs'],
        option['num_steps'],
        option['learning_rate'],
        option['proportion'],
        option['seed'],
        option['lr_scheduler']+option['learning_rate_decay'],
        option['weight_decay'],
        option['net_drop'],
        option['net_active'])
    return output_name

class Logger:
    def __init__(self):
        self.output = collections.defaultdict(list)
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf={}

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key = ''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key = ''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event.
            创建结束事件“key”的时间戳，并打印事件的时间间隔 """
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] =  time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(key+":", self.time_buf[key][-1]) + 's')
            # print("{:<30s}{:.4f}".format(key + ":", self.time_buf[key][-1]) + 'sss')

    def save(self, filepath):
        """Save the self.output as .json file"""
        if len(self.output)==0: return
        with open(filepath, 'w') as outf:
            ujson.dump(dict(self.output), outf)
            
    def write(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name==None: raise RuntimeError("Missing the name of the variable to be logged.")
        self.output[var_name].append(var_value)
        return

    def log(self, server=None):
        pass

from sklearn.metrics import confusion_matrix, classification_report
class DefaultLogger(Logger):
    def __init__(self):
        super(DefaultLogger, self).__init__()

    def log(self, server=None, current_round=-1):
        if len(self.output) == 0:
            self.output['meta'] = server.option
        test_metric = server.test()

        y_predict = test_metric['y_predict']
        del test_metric['y_predict']
        y_true = test_metric['y_true']
        del test_metric['y_true']
        pre = classification_report(y_true, y_predict, digits=4)
        print(pre)  # pre, re, f1
        con_mat = confusion_matrix(y_true, y_predict, labels=[0, 1, 2, 3, 4]) # , 5, 6, 7, 8, 9])
        # con_mat = confusion_matrix(y_true, y_predict, labels=[0, 1, 2, 3])
        print(con_mat)  # 输出混淆矩阵

        far_list = []
        ar_list = []
        acc_list = []
        for i in range(5):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            # acc1 = (tp + tn) / number
            ar1 = tp / (tp + tn)
            far1 = fp / (fn + fp)
            acc1 = (tp + tn) / (tp + tn + fp + fn)
            # acc_list.append(acc1)
            ar_list.append(ar1)
            far_list.append(far1)
            acc_list.append(acc1)
        print("ar:", ar_list)  # 检测率
        print("acc:", acc_list)
        print("far:", far_list)  # 虚报率



        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # calculate weighted averaging of metrics of training datasets across clients
        # train_metrics = server.test_on_clients(self.current_round, 'train')
        # for met_name, met_val in train_metrics.items():
        #     self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.client_vols, met_val)]) / server.data_vol)
        # calculate weighted averaging and other statistics of metrics of validation datasets across clients
        valid_metrics = server.test_on_clients(self.current_round, 'valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.client_vols, met_val)]) / server.data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        for key, val in self.output.items():
            if key == 'meta': continue
            print(self.temp.format(key, val[-1]))
