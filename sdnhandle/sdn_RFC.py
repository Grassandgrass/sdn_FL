from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

flow_dataset = pd.read_csv('FlowStatsfile.csv')

X_flow = flow_dataset.iloc[:, :-1].values
X_flow = X_flow.astype('float64')

y_flow = flow_dataset.iloc[:, -1].values

X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.4,
                                                                                random_state=0)

classifier = RandomForestClassifier(n_estimators=160,
                                    max_depth=15,
                                    criterion="entropy",
                                    n_jobs=-1)# n_estimators=10, criterion="entropy", random_state=0)
flow_model = classifier.fit(X_flow_train, y_flow_train)

y_flow_pred = flow_model.predict(X_flow_test)

print("------------------------------------------------------------------------------")
print("confusion matrix")
cm = confusion_matrix(y_flow_test, y_flow_pred)
print(cm)

acc = accuracy_score(y_flow_test, y_flow_pred)

print("succes accuracy = {0:.2f} %".format(acc * 100))
fail = 1.0 - acc
print("fail accuracy = {0:.2f} %".format(fail * 100))
print("------------------------------------------------------------------------------")




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
    global X_flow_train, X_flow_test, y_flow_train, y_flow_test

    ##  print(Y_train_encode.shape) # (336803,)

    rf = RandomForestClassifier(n_estimators=n_estimators_value,
                                max_depth=max_depth_value,
                                n_jobs=-1) # 处理器内核数量
    rf.fit(X_flow_train, y_flow_train)  # 训练分类器
    pred_y = rf.predict_proba(X_flow_test)
    ## print(len(pred_y), len(Y_test_encode)) # 22544 22544

    roc_auc = metrics.roc_auc_score(y_flow_test, pred_y[:,1]) # , multi_class='ovr')
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
