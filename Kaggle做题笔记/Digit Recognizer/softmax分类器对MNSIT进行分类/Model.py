import numpy
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim
from softmax分类器对MNSIT进行分类.ReadInData import trainDataset, predictDataset

# 定义超参数
"""
learning_rate : 学习率
batch_size_train : 训练集每批次训练数据量（每次训练行数）
batch_size_test : 测试集每批次测试数据量（每次测试行数）
epochs : 训练周期
momentum : 冲量，优化器参数。
"""
learning_rate = 0.01
batch_size_train = 64
batch_size_test = 1
epochs = 3
momentum = 0.5
log_interval = 10
random_seed = 1
batch_size_predict = 1
torch.manual_seed(random_seed)

# 获取数据集中训练数据并初始化数据加载器train_loader
dataset = trainDataset()
# 将dataset划分为训练集和验证集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size_train)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size_test)
# 创建预测集加载器
predict_dataset = predictDataset()
predict_loader = DataLoader(predict_dataset, shuffle=False, batch_size=batch_size_predict)


# 构建模型(Softmax分类器)
class Model(torch.nn.Module):
    def __init__(self):
        # 构建基本的输入层、隐藏层、输出层
        super(Model, self).__init__()
        # 由于将数据一维化后每行只有784列，即每行只有784个特征, 故输入为784
        self.linear1 = torch.nn.Linear(784, 528)
        self.linear2 = torch.nn.Linear(528, 384)
        self.linear3 = torch.nn.Linear(384, 256)
        self.linear4 = torch.nn.Linear(256, 128)
        self.linear5 = torch.nn.Linear(128, 64)
        self.linear6 = torch.nn.Linear(64, 10)


    def forward(self, x):
        # forward前馈计算主要用于计算预测值
        # 需要用到激活函数，由于损失函数选用交叉熵CrossEntropy()所以最后一层输出层无需进行激活(选取relu作为激活函数)
        y_predict = F.relu(self.linear1(x))
        y_predict = F.relu(self.linear2(y_predict))
        y_predict = F.relu(self.linear3(y_predict))
        y_predict = F.relu(self.linear4(y_predict))
        y_predict = F.relu(self.linear5(y_predict))
        return self.linear6(y_predict)

# 开始对模型进行训练
model = Model()
# 定义损失函数和权重优化器
# 损失函数选用CrossEntropy
criterion = torch.nn.CrossEntropyLoss()
# 优化器选用随机梯度下降SGD
# 优化器参数为    1. 待更新参数 2. 超参数 3. 冲量
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=momentum)


# 模型训练函数

def train(Epoch_Count):
    train_losses = []
    model.train()
    running_loss = 0
    for epoch in range(Epoch_Count):
        print(f"Epoch : {epoch + 1}")
        for batch_idx, (inputs, label) in enumerate(train_loader, 0):
            # 对模型进行训练
            # 先对梯度进行清零
            optimizer.zero_grad()
            # 前馈操作
            Y_predict = model(inputs)
            loss = criterion(Y_predict, label)
            # 反向传播
            loss.backward()
            # 优化器优化权重
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 200 == 0:
                print("[Training Epoch : %d, Batch : %5d / %d] loss: %.9f" % (epoch + 1, batch_idx, len(train_loader), running_loss/300))
                train_losses.append(running_loss)
                running_loss = 0.0
        if epoch % 1 == 0:
            test()
    return train_losses



# 模型测试函数
"""
    输入：验证集的数据（验证集数据由最初的训练集数据分割而来）
    功能：由于验证集是标签已知且未用来训练模型的数据，故用验证集来测试模型的泛化能力。
"""
test_accuracy = []
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (image, label) in test_loader:
            label_predict = model(image)
            # 取预测中每行最大值下标作为预测结果 torch.max()函数返回最大值的索引，对应的就是label
            _, predicted = torch.max(label_predict.data, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum()
    test_accuracy.append(100 * correct /total)
    print(f"Accuracy is {100 * correct /total}")



# 为训练的损失进行绘图
"""
输入 ：输入一个损失序列
输出 ：损失序列随时间（即随训练周期epoch进行的迭代曲线）
"""
def Digitplot(train_loss, test_accuracy):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    loss_indice = [i for i in range(len(train_loss))]   # loss_indice作为loss序列的索引，每两个算一个周期
    accuracy_indice = [i for i in range(len(test_accuracy))]

    # 初始化整个画板
    fig = plt.figure()
    # 初始化子图分布
    ax1 = fig.add_subplot(2, 1, 1)  # 参数表示：（row, col, num）
    ax2 = fig.add_subplot(2, 1, 2)
    # 绘制第一幅图
    """
    x轴：时间序列，训练周期 loss_indice
    y轴：损失序列，train_loss
    """
    ax1.plot(loss_indice, train_loss, label="训练损失")
    plt.xlabel("训练周期")
    plt.ylabel("训练损失")
    # 绘制第二幅图
    """
    x轴：时间序列，测试周期 accuracy_indice
    y轴：精度序列，test_accuracy
    """
    ax2.plot(accuracy_indice, test_accuracy, label="验证精度")
    plt.xlabel("测试周期")
    plt.ylabel("测试精度")

    plt.show()


# 预测函数
def Predict(Predict_Loader):
    Labels = []
    with torch.no_grad():
        for image in Predict_Loader:
            label_predict = model(image)
            # 取预测中每行最大值下标作为预测结果 torch.max()函数返回最大值的索引，对应的就是label
            # torch.max(, dim = 1)即返回每行的最大值的值和在当前行的索引
            _, predicted = torch.max(label_predict.data, dim=1)
            for label in predicted.numpy().tolist():
                Labels.append(label)
    return Labels




if __name__ == "__main__":
    train_loss = train(17)
    Digitplot(train_loss, test_accuracy)
    Predict_labels = Predict(predict_loader)
    print(f"Predict = {Predict_labels}")
    indice = [i for i in range(len(Predict_labels))]
    with open("../Predict result.csv", "w") as fp:
        fp.write("ImageId,Label\n")
        for i in range(len(indice)):
            fp.write(f"{indice[i] + 1} , {Predict_labels[i]}\n")
