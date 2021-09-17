# Readme

Date 2021-08-03

Made By 纸 石头 紫阳花

______



[TOC]

## 如何确定卷积神经网络参数

关于如何确定CNN中的卷积核大小、卷积层数以及每层的feature map个数，目前还没有一个理论性很强的解释，跟多的实在根据已有的经验设计或者利用自动搜索的方法搜索出较为合适的取值。

但也有一些比较普适的准则。

### 卷积核大小

理论上来说，卷积核大小是任意的，但是绝大部分情况下选取的是==奇数大小正方形卷积核==，且一般来说使用维度不大于7的卷积核（3x3、5x5、7x7），因为多个小卷积核比一个大卷积核拥有更多的非线性变换，这使得CNN对特征的学习能力更强。

在通用的神经网络中，大多是使用一个7x7的大卷积核，随后使用3x3卷积进行堆叠，在轻量化的神经网络中，则大多是全部采用3x3的卷积核。



### 卷积层数

早期最为经典的几个CNN，如AlexNet、VGGNet、GoogleNet、ResNet网络层数是不断加深的，在BatchNorm和残差结构出现后，以前深层神经网络容易出现的梯度消失、难以训练的问题得到缓解，从而使网络层数可以加到非常深。

目前来说，轻量级的神经网络层数一般在几十层左右，儿较大的神经网络也很少有超过200层。

卷积层数乐队，带来的非线性拟合能力越强，即能识别的团复杂度越高，卷积层内卷积的神经元越多，提取目标的细节越丰富。



### feature map

大部分网络都遵循一个原则即当输出特征图尺寸减半是输出的特征图的通道数应当加倍，从而保证相邻的卷积层包含的信息差不会相差太大。

现有分类网络最后一层几乎都是使用softmax函数进行激活，输出的图片属于每一个类别的概率值，所以最后一层的节点数一定等于待分类图像的类别数。而在==全局平均池化（GAP）== 提出后，主流的网络几乎都是在特征图尺寸降低到7*7左右时，直接用GAP + 全连接 + softmax 输出类别概率。



## 卷积神经网络内部原理

### 卷积层进行互相关运算代码实现

```python
def corr2d(X, K):
    h, w = K.shape
    # Y是初始化的输出层，待卷积核对X运算后进行填充
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 循环表用用卷积核K对输入层进行互相关运算
    for i in range(Y.shape[0]):	# 遍历rows
        for j in range(Y.shape[1]):	# 遍历cols
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()	#sum()相当于对通道进行合并
    return Y
```



###  定义卷积层模型

基于定义的corr2d函数实现二维卷积层，在构造函数中将weight和bias声明为模型参数。

```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))	# 未经训练的权重是随机定义的,在训练中逐周期进行更新
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return corr2d(x, self.weight) + bias
    
```



### 学习由X生成Y的卷积核

通过内置的二维卷积层并忽略偏置进行构造

```python
conv2d = nn.Conv2d(1, 1, kernel_size= (1, 2), bias=False)	# 构建一个输入输出都为1个通道且形状为（1， 2）的二维卷积层。
# conv2d使用四维的输入输出格式（批量大小，通道， 高度， 宽度）
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_pre = conv2d(X)
    loss = (Y_pre - Y)**2
    conv2d.zero_grad()
    l.sum().backward()	# 链式计算梯度
    # 迭代卷积核
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad	# 3e-2是学习率，这里相当于是优化器对参数进行更新操作
    if (i + 1) % 2 = 0:
        print(f"batch{i + 1}, loss {l.sum():.3f}")
        
```



严格卷积运算: $(f*g)(i, j) = \sum_{a}\sum_{b}f(a, b)g(i-a, j-b)$​ 

互相关运算转换为严格卷积运算只需要水平和垂直反转二维卷积张量，然后对输入张量进行互相关运算即可。

但是由于卷积核（对于卷积核张量上的权重称为==元素==）是从数据中学习到的，因此无论这些层执行严格的卷积运算还是互相关运算，最后的结果都是一样的，将得到相同的输出（因为初始的卷积核数据本身就是随机的），故一般直接将互相关运算作为卷积层学习的算法。

学习得到的卷积核被称为==特征映射（Feature Map）==因为它可以被视为一个输入映射到下一层的空间维度的转换器。在CNN中对于某一层中的任意元素x，其反向传播中可能影响到x计算的所有元素称为==感受野==。

### 多输入输出通道

#### 多输入通道

当输入包含多个通道时需要构造一个与输入通道具有相同输入通道数目的卷积核。

##### 多输入通道的互相关运算

```python
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))	# 遍历X, K的每一个维度进行卷积运算后将所有输出相加。
```

#### 多输出通道

在神经网络架构中，随着神经网络层数的加深，通常会增加输出通道的维度从而通过减少空间分布率以获取更大的通道深度。即，我们可以将每个通道看作是特征的响应。为了获取多通道的输出，我们需要为每一个通道配置一个三维卷积核。

#### 多通道输出的互相关运算

```python
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)	# torch.stack()函数用于将通道进行连接，可以将矩阵按时间序列压缩成一个矩阵，0指的是压缩的维度dim。
```



利用多通道输入输出的原理，可以通过1x1卷积核来改变数据的通道数量

```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape		# 输入数据的通道数、宽、高
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))	# 输入数据
    K = K.reshape((c_o, c_i))	# 卷积核
    Y = torch.matmul(K, X)
    return Y.shape(c_o, h, w)
```



### 汇聚层（池化层）

#### 池化层的目的

降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。

#### 池化层基本原理

池化层不改变输入的通道数，而是由一个类似卷积核的mxn窗口对输入的某一个通道进行遍历，且窗口在遍历过程中保持不重叠。通常我们选取当前窗口中最大值（==最大池化层==）或者平均值（==平均池化层==）进行拼接形成输出，从而降低一两个数据变动所带来的敏感性。

```python
def pool2d(X, pool_size, mode="max"):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h, X.shape[1] - p_w + 1))
    for i in range(Y.shpe[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
	return Y

```

默认情况下，深度学习框架中池化层的步幅大小与池化窗口大小保持一致`pool2d = nn.MaxPool2d(3)`。但也可以通过参数padding和stride 进行二次确定。



## 经典卷积神经网络模型

### 卷积神经网络LeNet

LeNet时最早发布的神经网络之一，主要由两个部分组成：

1. 卷积编码器：由两个卷积层组成
2. 全连接密集块：由三个全连接层组成

每个卷积块的基本单元是一个卷积层，一个激活函数和池化层。为了将卷积块的输出传递给稠密块，需要在小批量中展平每个样本，由四维输入(batch(批量大小), channel, height, weight)转换成全连接层所期望的二维输入（索引，每个样本的平面向量表示）。

#### PyTorch 对LeNet模型进行实例化

```python
class Sequential(torch.nn.Module):
    def __init__(Sequential, self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.avgpool = torch.nn.AvgPool2d(2)
        self.sigmoid = torch.nn.sigmoid()
        self.flatten = torch.nn.Flatten()	# 卷积层与线性层之间需要展平层来进行过度
        self.linear1 = torch.nn.Linear(16*(5*5), 120)# 16*(5*5)是卷积层的输出(通道数 * 宽度 * 高度)的展平结果
# 宽高为5的前提是输入的是宽高为32像素的图像，32->卷积->32-4=28->下采样->28/2=14->卷积->14-4=10->下采样->10/2=5
        self.linear2 = torch.nn.Linear(120, 84)
        self.linear3 = torch.nn.Linear(184, 10)       
		        
	def forward(self, x):
        x = self.avgpool(self.sigmoid(self.conv1(x)))
        x = self.avgpool(self.sigmoid(self.conv2(x)))
        x = self.flatten(x)
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.linear3(x)
		return x.view(-1, 1, 28, 28)

net = Sequential()

```

##### 对模型精度进行评估

```python
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
		net.eval()		# 设置为评估模式
        if not device:
            device = next(iter(net.paramters())).device
    metric = d2l.Accumulator(2)
    for X, y in data_iter():
        if isinstance(X, list):
	        X = [x.to(device) for x in X]
    	else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(x), y), y.numel())	# numel()用于返回元素个数
    return metric[0]/metric[1]

```

Accumulator 是一个使用程序类(自定义，非系统类)，用于存储正确预测的数量和预测的总数量

##### 对模型进行训练

```python
def train(net, train_iter, test_iter, num_epochs, lr, device):
    dnet.apply(init_weights)
    print(f"training on {device}")
    net.to(device)
    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropy()
    # 继承绘图类
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs], legend=["train_loss", "train_accuracy", "test_accuracy"])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 继承精度计算类
        metric = d2l.Accumulator(3)
        # 进入训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            # 将待计算数据存入GPU
            X, y = X.to(device), y.to(device)
            # 开始正向传播
            y_predict = net(X)
            loss = criterion(y_predict, y)
            # 开始反向传播
            loss.backward()
            # 优化器对权重进行更新
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0], d2l.accuracy(y_predict, y), X.shape[0])
            timer.stop()
            train_loss = metric[0]/metric[1]
            train_accuracy = metric[1]/metric[2]
            # 绘图
            # 训练集损失程度
            if (i + 1) % 5 (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_arr, None))
            # 验证集精度
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
            print(f"loss {train_l:.3f}, train acc {train_acc:.3f}, test_acc {test_acc:.3f}")
            print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}")
            
```

