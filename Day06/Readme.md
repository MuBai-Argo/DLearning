# Readme

Date:2021-07-31

Made By: 纸 石头 紫阳花

________________





[TOC]



## Pytorch 如何对不同格式的数据集进行预处理



### Tensor与Numpy的转换

Tensor和Numpy在转换前后占用的可以是同一块内存

1. Tensor转换为Numpy数组<br>
    ```python
    n = t.numpy()
    ```
    通过张量的.numpy()方法由Tensor转换为Ndarray。

2. numpy转换为Tensor

   ```python
   t = torch_from_numpy(n)
   ```

    

### 从DataFrame中传数据到神经网络模型

首先从文件中读取DataFrame类型的数据，利用iloc方法将标签与数据分离。随后得到数据和标签的DataFrame形式。

通过DataFrame数据类型的.values方法可以将DataFrame数据用numpy的ndarry进行表示。从而可以使用torch_from_numpy（）方法将ndarray形式数据转换为Tensor形式。

有了Tensor形式的数据与标签后，开始构建Dataset的子类，Dataset的子类需要重载两个基本魔术方法==\__len__==和 ==\__getitem__== 

在实例化该类对象后，利用`torch.utils.data.random_split`方法 将数据集分解为训练集和验证集，分别用`torch.utils.data.DataLoader` 初始化训练集和验证集的数据加载器，用于训练模型和测试模型泛化能力。

#### 代码实现

```python
# 创建数据集
class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        train_dataset = pd.read_csv("../train.csv")
        test_dataset = pd.read_csv("../test.csv")

        X_train = train_dataset.iloc[:, 1:]
        Y_train = train_dataset.iloc[:, 0]
        
        X_train = torch.from_numpy(X_train.values)
        Y_train = torch.from_numpy(Y_train.values)

        self.len = X_train.shape[0]
        self.X_train = X_train.float()
        self.Y_train = Y_train.long()

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.len

# 获取数据集中训练数据并初始化数据加载器train_loader
dataset = MyDataset()
# 将dataset划分为训练集和验证集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size_train)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size_test)

```





## Pytorch 中net.train和net.eval的使用

PyTorch框架下在训练模型是会在前面加上`modedl.train()` ，在测试模型时会在前面使用`model.eval()`

在实际运行中，这两段代码不写也可以运行，这两个方法是针对在网络训练和测试时采用不同方式的情况，比如 `Batch Normalization` 和 `Dropout` 





## PyTorch 切分数据集



torch.utils.data是PyTorch关于数据集操作的文件

### data 文件中的常用类与函数

- `Dataset` 抽象类，其他所有类的数据集类都应该是其子类，而且其子类必须重载两个重要函数

    1. `__len__(self)` : 提供数据集大小

    2. `__getitem__` : 支持整数索引

- `TensorDataset` 将numpy数据集封装成Tesnor类型的数据集`Dataset` 类型。
- `ConcatDataset` 连接不同的数据集以构成更大的新数据集
- `Subset(dataset, indices)` 获取指定一个索引序列以构成更大的新数据集
- `DataLoader(dataset, batch_size=1 ,shuffle=False, sampler=None, batch_sample=None, num_workers=0, collate_fn=<functinon default_collate>, pin_memory=False, drop_last=False, timeout=0, work_init_fn = None)` 数据加载器，组合一个数据集与提样器，用于将数据集迭代化并根据批次提供数据。
-   `random_split(dataset, lengths)` 按照给定的长度将数据集划分为没有重叠的新数据集组合
- `Sampler(data_source)` 所有采样器的基类， 每个采样器子类都需要提供iter方法以方便迭代器进行所有和一个len方法返回迭代器长度。
- `SequentialSampler(data_source)` 顺序采样样本
- `RandomSampler(data_source)` 不放回随机采样样本元素
- `SubsetRandomSampler(indice)` 不放回按照给定所应列表进行采样
- `WeightedRandomSampler(weights, num_samples, replacement=True)` 按照给定概率采样
- `BacthSampler(smapler, batch_szie, drop_last)` 在一个batch中封装一个其他的采样器
- `DistributedSampler(dataset, num_replicas=None, rank=None)`  采样器可以约束数据加载进数据集的子集



### 分别利用 SubsetRandomSampler 和 random_split 对数据集进行分割

#### SubsetRandomSampler

```python
# 创建数据集
dataset = MyCustomDataset(my_path)
batch_size = 16
validation_split = 0.2	# 分割比例
shuffle_dataset = True
random_seed = 42

# 为分割数据创建索引
dataset_size = len(dataset)
indices = list(range(dataset_size))
# 确定分割比例
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# 创建数据采集器和数据加载器
# 构建数据采集器
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
# 构建数据加载器
train_loader = DataLoader(dataset, batch_size=batch_size, sampler = train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

```

##### 数据采样器和数据加载器

在PyTorch中，Sampler决定读取数据时的先后顺序，DataLoader负责装在数据并更具Sampler提供的数据安排数据。

默认DataLoader采用顺序采样器（shuffle = True）或者随机采样器（random sampler）

1. 首先Sampler会根据Dataset的大小n形成一个可迭代的序号列表[0, n-1]
2. BatchSampler根据DataLoader的batch_size参数将Sampler提供的序列划分为多个batch大小的可迭代序列组，drop_last参数决定是否保留最后一组（不足batch_size）。
3. Sampler和Dataset合二为一，在迭代读取DataLoader时，用BatchSampler中的一个batch编号去查找Dataset中对应的数据和标签，读出一个batch数据。

#### random_split

```python
train_size = int((1-validation_split) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
```



### 对比sklearn的数据集划分方法

```python
from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target = train_test_split(li.data, li.target, test_size = 0.25)
```





## 对模型的训练过程进行评估

### 如何在训练周期中对训练损失函数的计算结果进行表示





### Python 绘图











## 在深度学习中如何正确的选择激活函数



常用的激活函数包括==Sigmoid== 、==Relu==、 ==Gelu==、 ==Tanh== 、==ELU==、==Leaky Relu== 、==SELU==

 选择激活函数时要考虑到梯度可能产生的问题

### 梯度问题

#### 梯度消失

梯度很小，如同消失了一样时会导致神经网络中的权重几乎没有更新，导致网络中节点与最优值相差较大，陷入鞍点。

#### 梯度爆炸

解决梯度爆炸的问题的基本思路就是为其设定一个规则，选择一个阈值，假如梯度超过这个值，则使用==梯度裁剪==或==梯度规范==。

### 常用激活函数

针对梯度消失问题，一般采用==Relu函数==作为激活函数。当引入了Relu函数时，我们向神经网络中引入了很大的稀疏性，这意味着激活的矩阵中包含着很多的0。这能很大的提高时间和空间复杂度方面的效率（因为常数值通常需要的空间更少计算成本更低），但是Relu函数能解决梯度消失问题，却不能解决==梯度爆炸==问题。

==ELU函数== 即指数线性函数的图像与Relu函数接近，由于指数函数的图像特征，在x < 0的情况下，图像以$\alpha(e^x - 1)$ 进行非线性分布。$\alpha$ 为超参数。由于引入了指数运算，所以ELU的计算成本比Relu高。由于$\alpha(e^x - 1)$ 的函数分布所以输入值无法映射到很小的输出值，所以梯度小事问题可以很好的得到解决。ELU的梯度恒大于0故可以避免==死亡Relu==（大量系欸但梯度为0）的情况，到那时计算周期更长，也无法避免梯度爆炸问题。$\alpha$ 值需要通过手动调参凭经验确定。

==Leaky Relu== 即渗漏型整流线性单元激活函数，函数为

$$\begin{equation} LReLU(x) =  \begin{cases} x & \mbox{if x > 0}\\ \alpha x&\mbox{if x <= 0}\end{cases} \end{equation} \space\space\space\space\space \alpha的取值一般在0.01到0.03之间$$<br>相当于是ELU和Relu函数之间的折中选择。

==SELU== 即拓展型指数线性单元，在应用中需要使用lecun_normal进行权重初始化，当需要dropout是，需要使用AlphaDropout。
SELU的公式为 $$\begin{equation} SELU(x) = \lambda \begin{cases} x & \mbox{if x > 0}\\ \alpha (e^x - 1)&\mbox{if x <= 0}\end{cases} \end{equation} \space\space\space\space\space \alpha和\lambda都是确定值，由模型提供 $$​ SELU的输出是内部归一化的加快了网络的收敛速度，同时也避免了梯度消失和梯度爆炸的问题。

==GELU== 即高斯误差线性单元 常用于NLP自然语言处理，能避免梯度消失的情况。





