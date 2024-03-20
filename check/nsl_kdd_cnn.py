import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import time


from benchmark.nslkdd_muti_classification.nsl_mutil_kdd import load_data
# from check.LMNet_lstm import Model
from check.cnn import Model
from check.nslkdd.nslkdd_Dataset import nslkdd_Dataset


batch_size = 64
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# train_set = datasets.MNIST(root='../dataset/minist', train=True, transform=transform, download=True)
# train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
# test_set = datasets.MNIST(root='../dataset/minist', train=False, transform=transform, download=True)
# test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)


trainx, trainy, testx, testy = load_data()  # train shape: (82332, 196) ; train_label shape:  (82332, 1)

train_set = nslkdd_Dataset(trainx, trainy)
train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
test_set = nslkdd_Dataset(testx, testy)
test_loader = DataLoader(test_set, shuffle=False, batch_size=512)

epoch_list = []
accuracy_list = []


# 模型构建
model = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 用显卡来算
model.to(device)

# 构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss(
            # weight=torch.from_numpy(np.array([0.05, 0.6, 0.6, 2.0, 2.0])).float().cuda(),
            # weight=torch.from_numpy(np.array([0.1, 0.8, 0.8, 2.0, 1.2])).float().to(device), # 75.691 %
            weight=torch.from_numpy(np.array([0.1, 0.6, 0.6, 2.0, 1.2])).float().to(device),  # 0.4 * con1-2 0.7935 %
            size_average = True,
        )
# 选择交叉熵损失函数，那么就包括了softmax函数和NLLLoss过程，即最后一层不用激活函数激活
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0) # lr=0.05


# 训练循环（前向，反向以及更新）
def train(epoch):
    running_loss = 0.0
    # enumerate函数表示将遍历数据，并将其组成一个索引序列，0表示索引从0开始以元组形式返回索引序列中的值。
    for batch_index, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        # 之前的梯度清零
        optimizer.zero_grad()
        # 获得模型预测结果（64*10）# print(inputs.shape) # torch.Size([64, 1, 14, 14])

        outputs = model(inputs.float())
        # 交叉熵损失函数outputs(64*10)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 2000))
            running_loss = 0.0

from sklearn.metrics import confusion_matrix, classification_report
def mstest():
    correct = 0
    total = 0

    targets_list = []
    predicted_list = []
    # 测试阶段不存在反向传播，所以不需要进行梯度求导
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            # 列是0维，行是1维，即取1维条件下的最大输出值
            _, predicted = torch.max(outputs.data, dim=1)  # 由于torch.max返回的是最大值和索引值,predicted需要的是索引值，'_,'是最大值的位置
            total += targets.size(0)  # 将测试集中（N*1）的矩阵中，第0个元素进行取出，最后得到测试集的测试数目
            # 预测值和测试的真实值进行对比，如相等，则将其数据提取出来，并相加
            correct += (predicted == targets).sum().item()

            targets_list = targets_list + targets.cpu().numpy().tolist()
            predicted_list = predicted_list + predicted.cpu().numpy().tolist()
    print('accuracy on test set: %.3f %%' % (100 * correct / total))

    pre = classification_report(targets_list, predicted_list, digits=4)
    con_mat = confusion_matrix(targets_list, predicted_list, labels=[0, 1, 2, 3, 4])  # , 5, 6, 7, 8, 9])
    print(pre)
    print(con_mat)

    return correct / total


if __name__ == '__main__':
    for epoch in range(50):
        torch.cuda.synchronize()
        start = time.time()
        train(epoch)
        torch.cuda.synchronize()
        end = time.time()
        total_time = end - start
        print('total_time:{:.2f}'.format(total_time))

        acc = mstest()
        epoch_list.append(epoch)
        accuracy_list.append(acc)
    plt.plot(epoch_list, accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
