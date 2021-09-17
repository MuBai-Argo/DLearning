import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim

# 创建数据集
class trainDataset(Dataset):
    def __init__(self):
        super(trainDataset, self).__init__()
        train_dataset = pd.read_csv("../train.csv")

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


# 预测集（测试集）
class predictDataset(Dataset):
    def __init__(self):
        super(predictDataset, self).__init__()
        predict_dataset = pd.read_csv("../test.csv")

        X_predict = torch.from_numpy(predict_dataset.values)

        self.len = X_predict.shape[0]
        self.X_predict = X_predict.float()


    def __getitem__(self, index):
        return self.X_predict[index]


    def __len__(self):
        return self.len


