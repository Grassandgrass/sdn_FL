import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from benchmark.unswnb15_muti_classification.data_mutil_process import load_data
# from check.LMNet_lstm import Model
from check.scnn import Model
from check.un15datesets import UN15Dataset

# 准备数据
# transform.Compose（转换操作包括），其中是个列表，所以会遍历该列表，并执行
# transform.ToTensor 输入模式为PIL Image 或 numpy.ndarray (形状为H x W x C)数据范围是[0, 255] ，转换到一个 Torch.FloatTensor，
# 其形状 (C x H x W) 在 [0.0, 1.0] 范围
# transform.Normalize 是指归一化操作，其中均值和方差为0.1307和0.3081


batch_size = 64
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# train_set = datasets.MNIST(root='../dataset/minist', train=True, transform=transform, download=True)
# train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
# test_set = datasets.MNIST(root='../dataset/minist', train=False, transform=transform, download=True)
# test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)


trainx, trainy, testx, testy = load_data()  # train shape: (82332, 196) ; train_label shape:  (82332, 1)

train_set = UN15Dataset(trainx, trainy)
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_set = UN15Dataset(testx, testy)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

epoch_list = []
accuracy_list = []


# 模型构建
model = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 用显卡来算
model.to(device)

# 构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 选择交叉熵损失函数，那么就包括了softmax函数和NLLLoss过程，即最后一层不用激活函数激活
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0)    # lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0 # 增加了惯性冲量，有利于克服鞍点


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


def mstest():
    correct = 0
    total = 0
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
    print('accuracy on test set: %d %%' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    for epoch in range(500):
        train(epoch)
        acc = mstest()
        epoch_list.append(epoch)
        accuracy_list.append(acc)
    plt.plot(epoch_list, accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
