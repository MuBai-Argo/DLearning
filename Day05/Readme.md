# Readme

  Date:2021-07-30
	    Made By: 纸 石头 紫阳花

****************




[TOC]

# RNN循环神经网络

> md前面学的都忘了

## RNN基本概念

RNN基本原理 线性层复用

RNN神经网络专门用来处理带有序列模式的数据，其中也需要使用权重共享的概念来减少需要训练的权重的数量。

在节点间不仅要考虑到节点的连接关系，而且要考虑到各节点之间是具有一种时间序列的关系，即后序节点将部分依赖于前序节点的状态。

故RNN主要用于处理具有序列连接的输入数据，诸如天气数据、金融数据、自然语言。

### RNN Cell

当我们接受到序列当中时刻t的数据，我们把m维的数据通过RNN Cell变为n维向量。

RNN Cell的本质是利用同一线性层处理按照时间序列分布的多个数据，将某个维度的向量映射到另一个维度的空间中。但RNN Cell相对于一般线性层来说是共享的，即$x_1$ 经过RNN Cell输出隐藏层的$ h_1$ 仅包含$x_1$ 中的信息,$x_2$ 经RNN Cell输出的隐藏层$h_2$ 不仅包含$x_2$ 信息还包含$x_1$ 信息，以此类推。即RNN Cell不经输出了$h_1$ 还在下一次运算中自动将$h_1$ 作为先验值输入了RNN Cell中。  

RNN Cell的使用减少了节点数量，并体现了数据间的依赖关系。

```python
RNN_Cell = Linear()
h = 0
for x in Xs:
	h = RNN_Cell(x, h)
    h = F.tanh(h)
```



在RNN 神经网络中一般使用tanh函数作为激活函数。



## 构造RNN神经网络

在PyTorch中有两种构造RNN神经网络的方式

1. 通过构造RNN Cell, 然后自己构造循环

    ```python
    import torch
    
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2
    
    cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)	# 参数为输入维度和隐层维度
    dataset = torch.randn(seq_len, batch_size input_size)	# seq_len是序列的长度
    hidden = torch.zeros(batch_size, hidden_size)
    
    for idx, input in enumerate(dataset): 
    	hidden = cell(input, hidden)
    ```

2. 直接使用RNN模型
    input of shape (seqSize, batch, input_size)
    hidden of shape(numLayer, batch, hidden_size)
    output of shape(sqlSize, batch, hidden_size)

    ```python
    import torch
    
    batch_size = 1
    sql_len = 3
    input_size = 4
    hidden_size = 2
    num_layers = 1
    
    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    
    inputs = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    
    out, hidden = cell(inputs, hidden)
    
    print(f"Output size = {out.shaoe}")
    print(f"Output = {out}")
    print(f"Hidden size = {hidden.shape}")
    print(f"Hidden = {hidden}")
    
    ```

    

### RNN实例

文本转换“hello”->“ohlol”

首先用将文本内容转换为独热向量(e:0, h:1, l:2, o:3)$[\begin{bmatrix}0&1&0&0\\1&0&0&0\\0&0&1&0\\ 0&0&1&0\\0&0&0&1\end{bmatrix}]$​​​ 只有四个字符inputsize = 4 = outputsize，属于四分类问题。​

### RNNCell

#### 构建数据集

```python
import torch

input_size = 4
output_size = 4
batch_size = 1

idx2char = ["e", "h", "l", "o"]
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1],]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)

```

#### 构建模型

```python
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size)
    	super(Model, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        
    def forward(self, input, hidden):
		hidden = self.rnncell(input, hidden)
        return hidden
    
    def init_hidden(self)
		return torch.zeros(self.batch_size, self.hidden_size)
    
net = Modle(input_size, hidden_size, output_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = troch.optim.Adam(net.paramters(), lr=0.1)
```

#### 训练模型

```python
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print(f"Predict string = ", end="")
    for inputm label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end="")
       	loss.backward()
        optimizer.step()
        print(f",[{epoch+1}/15] loss = {loss.item()}")
```

###  RNN

#### 构建模型

```python
class Model(torch.nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers)
        
    def forward(self, input):
		hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return output.view(-1, self.hidden_size)
    
net = Module(input_size, hidden_size, batch_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = troch.optim.Adam(net.paramters(), lr=0.1)

```



#### 训练模型

```python
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(input)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    _, idx = output.max(dim=1)
    idx = idx.data.numpy()
	print(idx2char[idx.item()], end="")
	print(f",[{epoch+1}/15] loss = {loss.item()}")
    
```







### Embedding 嵌入层

把高维稀疏层的数据映射到低维稠密层中，即数据降维。



torch.nn.Embedding 初始化参数：

- 输入的独热向量维度 num_embeddings
- 构成矩阵的高度和宽度 embedding_dim

要求输入是一个长整数张量, 输出是input_shape和embedding_dim。

#### 网络结构

```python
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                              hidden_size=hidden_size,
                               num_layers = num_layers,
                               batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)
        
	def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)
```

 

