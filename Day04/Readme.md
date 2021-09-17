



# Readme

# Day04

 Date:2021-07-27
		Made By: 纸 石头 紫阳花

____

[TOC]

## 卷积神经网络进阶

### Inception

Inception模块用于对一段多次出现的神经网络进行封装,内部进行特征提取等操作。

#### Inception实现

构造卷积神经网络时有一些超参数比较难以抉择，所以我们可以在一个Inception块中把这些卷积都用一下，然后把结果拼接（每条路径的结果输出的宽度和高度都必须是一致的）在一起，将来增大最好用的卷积的权重，其他路线的权重就会变小，相当于是提供了几种候选的卷积神经网络的配置，然后通过训练自动找到最优的卷积组合。

```python
# 第一个分支：Average Pooling + 1x1 Conv 称为池化分支
self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)
# AveragePooling 池化层, 保持通道的宽高不变。
branch_pool = F.ave_pool2d(x, kernel_size=3, stride=1, padding=1)
branch_pool = self.brach_pool(branch_pool)
```

```python
# 第二个分支：1x1分支 仅包含一个1x1卷积层
self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

branch1x1 = slef.branch1x1(x)
```

```python
# 第三个分支：1x1Conv + 5x5Conv
self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
self.branch5x5_2 = torch.nn.Conv2d(16, 25, kernel_size=5, padding=2)

branch5x5 = self.branch5x5_1(x)
branch5x5 = self.branch5x5_2(branch5x5)
```

```python
# 第四个分支：1x1Conv + 3x3Conv + 3x3Conv
self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

branch3x3 = self.branch3x3_1(x)
branch3x3 = self.branch3x3_2(branch3x3)
branch3x3 = self.branch3x3_3(branch3x3)
```

```python
# Incepetion 拼接
output = [branch1x1, branch5x5, branch3x3, branch_pool]
return torch.cat(outputs, dim=1)
```

#### 完整的python代码实现Inception

```python
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
		super(Inception, self).__init__()
        # 第一个分支：Average Pooling + 1x1 Conv 称为池化分支
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)
        # 第二个分支：1x1分支 仅包含一个1x1卷积层
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        # 第三个分支：1x1Conv + 5x5Conv
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 25, kernel_size=5, padding=2)
        # 第四个分支：1x1Conv + 3x3Conv + 3x3Conv
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        
    def forward(self, x):
		branch1x1 = slef.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch_pool = F.ave_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.brach_pool(branch_pool)
        
        # Incepetion 拼接
        output = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)
        
        
```

##### 基于Inception构建神经网络

```python
class Net(torch.nn.Module):
	def __init__(self):
        super(Net, self).__init__()
        # 卷积核维度对输出层数据的影响 输出层宽高 = 输入层宽高 - （kernel_size - 1）
        # 参数为输入通道数和输出通道数
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)	
        self.conv2 = torch.nn.Conv2d(80, 20, kerner_size=5) 
        
        self.incep1 = InceptionA(in_channel=10)
        self.incep2 = InceptionA(in_channel=20)
        
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)
        
     def forward(self, x):				 
        batch_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))	
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(batch_size - 1)	# flatten
        x = self.fc(x)
        return x
    
model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optimizer.SGD(model.parameters(),
                          lr=0.01,
                          momentum=0.5)	 
```



##### 1x1卷积

表示卷积核尺寸为1x1，相当于对原始图像的每一个像素进行数乘。

1x1卷积最主要的工作在于通过设置不同数量的卷积核来改变通道的数量。

###### 改变通道的意义

卷积操作的运算量 = Input参与运算的通道数 x Output输出的通道数 x 卷积核参与运算数据量 x Input参与运算数据量

通过1x1卷积降低 Input参与运算的通道数 可以有效的降低卷积操作的运算量。实际上的运算数量减少了。



###### 梯度消失

由于反向传播需要将一连串的梯度相乘，假如每一处梯度都小于1，则权重的更新将将趋近于0。基于此原理，卷积层过多也可能导致训练拟合能力不足。

要解决梯度消失问题，需要加入==Residual net==，即跳连接。

经过两个权重层，先不进行激活，而是先与X（即两个权重层前的输入）进行相加后再进行激活。
$$
H(x) = F(x) + x
$$
由此可以解决梯度消失的问题，即确保每两次必有一次的梯度大于1，从而进行充分的训练。

如果x和F（x）的维度不一样，则需要进行单独处理，如过一个池化层转换为同样的大小。

实现

```python
class ResidualBlock(nn.Module):
	def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
	def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x)
```

加入Residual net的神经网络

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        
        self.fc = nn.Linear(512, 10)
        
    def forward(self, x)
    	in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x.view(in_size, -1)
        x = self.fc(x)
        return x
```



