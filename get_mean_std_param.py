import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import json
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import time
import tool as d2l
import pickle

#=================================
#cifar10
'''下载训练集 CIFAR-10 10分类训练集'''
# data_dir = '/Datasets/cifar-10/'
# batch_size = 64
# mini_batchsize = 32
# train_set,_,_,_ = d2l.load_cifar10_dataset(data_dir,batch_size=batch_size,transform=transforms.ToTensor())
# path = './cifar-10-norm.txt'
# mean,std = d2l.get_mean_std(train_set,path)
# print(mean)
# print(std)
# file = open(path,'rb')
# mean = pickle.load(file)
# std = pickle.load(file)
# print(mean,std)

#=================================
#FashionMnist
data_dir = '/Datasets/FsahionMNIST/'
batch_size = 64
mini_batchsize = 32
train_set,_,_,_ = d2l.load_fmnist_dataset(data_dir,batch_size=batch_size,transform=transforms.ToTensor())
path = './Fmnist-norm.txt'
mean,std = d2l.get_2C_mean_std(train_set,path)
print(mean)
print(std)
file = open(path,'rb')
mean = pickle.load(file)
std = pickle.load(file)
print(mean,std)