import matplotlib.pyplot as plt
import numpy as np

def zhuzhuangtu():
    # 有a/b/c三种类型的数据，n设置为3
    total_width, n = 0.8, 3
    # 每种类型的柱状图宽度
    width = total_width / n


    # 功能1
    x_labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    # 用第1组...替换横坐标x的值
    x = np.arange(5)
    plt.xticks(x, x_labels)
    # 重新设置x轴的坐标
    x = x - (total_width - width) / 2
    print(x)

    # a = [0.7947, 0.9619, 0.7966, 0.7332, 0.7586]
    # b = [0.9661, 0.8371, 0.8459, 0.4250, 0.1100]
    # c = [0.8721, 0.8952, 0.8205, 0.5381, 0.1921]
    a = [0.8428, 0.9456, 0.6909, 0.7196, 0.7586]
    b = [0.9657, 0.8077, 0.8273, 0.5493, 0.1100]
    c = [0.9001, 0.8712, 0.7530, 0.6230, 0.1921]

    # 功能2
    for i, j in zip(x, a):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
    for i, j in zip(x + width, b):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
    for i, j in zip(x + 2 * width, c):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)

    # 画柱状图
    plt.bar(x, a, width=width, label="Precision", color="white", edgecolor="k", hatch="/")
    plt.bar(x + width, b, width=width, label="Recall", color="white", edgecolor="k", hatch="***")
    plt.bar(x + 2*width, c, width=width, label="F1-score", color="white", edgecolor="k", hatch="xx")
    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()


def bijiao():
    # 准备数据

    # 正确显示中文和负号
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    x_labels = ["决策树", "随机森林", "朴素贝叶斯", "循环神经网络", "FL-SEResNet"]
    # 每种类型的柱状图宽度
    width = 0.5

    # 用第1组...替换横坐标x的值
    x = np.arange(5)
    plt.xticks(x)
    # 重新设置x轴的坐标

    # a = [0.7947, 0.9619, 0.7966, 0.7332, 0.7586]
    # b = [0.9661, 0.8371, 0.8459, 0.4250, 0.1100]
    # c = [0.8721, 0.8952, 0.8205, 0.5381, 0.1921]
    data = [74.60, 74.00, 74.40, 81.29, 84.01]

    # # 功能2
    for i, j in zip(x, data):
        plt.text(i, j, "%.2f" % j, ha="center", va="bottom", fontsize=13)
    # for i, j in zip(x + width, b):
    #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
    # for i, j in zip(x + 2 * width, c):
    #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)

    # 画柱状图
    plt.bar(x_labels, data, width=width)
    plt.ylim(0, 100)
    plt.ylabel("准确率%", fontsize=15)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()




test = zhuzhuangtu()




