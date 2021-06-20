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
import MyModels as M
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from KDBU_train import *

import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger('./log/test_vgg_model_log.txt')

def testCifarModel(meanstd_filepath,dataset_path,model,model_path):
    # 预处理标准化
    path = meanstd_filepath
    cifar_file = open(path, 'rb')
    mean = pickle.load(cifar_file)
    std = pickle.load(cifar_file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 获取数据
    data_dir = dataset_path #
    batch_size = 4
    train_set, _, train_iter, test_iter = d2l.load_cifar10_dataset(data_dir, batch_size=batch_size, transform=transform)
    #label = train_set.class_to_idx
    #===================================
    # 加载模型
    Model_path = model_path
    net = model #
    net.load_state_dict(torch.load(Model_path))
    net = net.to(device)
    net.eval()
    loss = nn.CrossEntropyLoss()
    d2l.eval_model(net, train_set, test_iter, loss, device)

def testFmnistModel(meanstd_filepath,dataset_path,model,model_path):
    # 预处理标准化
    path = meanstd_filepath
    cifar_file = open(path, 'rb')
    mean = pickle.load(cifar_file)
    std = pickle.load(cifar_file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 获取数据
    data_dir = dataset_path #
    batch_size = 4
    train_set, _, train_iter, test_iter = d2l.load_fmnist_dataset(data_dir, batch_size=batch_size, transform=transform)
    #label = train_set.class_to_idx
    #===================================
    # 加载模型
    Model_path = model_path
    net = model #
    net.load_state_dict(torch.load(Model_path))
    net = net.to(device)
    net.eval()
    loss = nn.CrossEntropyLoss()
    d2l.eval_model(net, train_set, test_iter, loss, device)
def testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize):
    # 预处理标准化
    mean = [ 0.5, 0.5, 0.5 ]
    std  = [ 0.5, 0.5, 0.5 ]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 获取数据
    data_dir = dataset_path #'/Datasets/cifar-10/'
    batch_size = batch_size
    mini_batchsize = mini_batchsize

    test_dataset, _, _ = d2l.load_trainval_dataset(data_dir, batch_size=batch_size, transform=transform)
    test_iter = DataLoader(test_dataset,shuffle=True,drop_last=True)
    #label = dataset.class_to_idx
    #===================================
    # 加载模型
    Model_path = model_path
    net = model #
    net.load_state_dict(torch.load(Model_path))
    net = net.to(device)
    net.eval()
    loss = nn.CrossEntropyLoss()
    d2l.eval_model(net, test_dataset,test_iter,loss, device)

def testCifar():
    #testCifarModel('./cifar-10-norm.txt','/Datasets/cifar-10/',M.get_CifarVGG16(),'./models/CifarVGG16.pt')
    #testCifarModel('./cifar-10-norm.txt','/Datasets/cifar-10/',M.get_CifarResNet(),'./models/CifarResNet.pt')
    testCifarModel('./cifar-10-norm.txt', '/Datasets/cifar-10/', M.get_CifarLeNet(), './models/CifarLeNet.pt')

def testFmnist():
    #testFmnistModel('./Fmnist-norm.txt', '/Datasets/FsahionMNIST/', M.get_MnistVGG16(),'./models/FmnistVGG16.pt')
    testFmnistModel('./Fmnist-norm.txt','/Datasets/FsahionMNIST/',M.get_MnistResNet(),'./models/FmnistResNet.pt')

def testFnet_fgsm():
    dataset_path = './datasets/cifar10/clean-fgsm/test/'
    model = M.get_FNet(num_classes=2)
    model_path = './models/Fnet_fgsm.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)
def testFnet_cw():
    dataset_path = './datasets/cifar10/clean-cw/test/'
    model = M.get_FNet(num_classes=2)
    model_path = './models/Fnet_cw.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)
def testFnet_dp():
    dataset_path = './datasets/cifar10/clean-dp/test/'
    model = M.get_FNet(num_classes=2)
    model_path = './models/Fnet_dp.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)
#=================================================
def testRGB_fgsm():
    dataset_path = './datasets/cifar10/clean-fgsm/test/'
    model = M.get_RGB(num_classes=2)
    model_path = './models/RGB_fgsm.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)

def testRGB_cw():
    dataset_path = './datasets/cifar10/clean-cw/test/'
    model = M.get_RGB(num_classes=2)
    model_path = './models/RGB_cw.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)
def testRGB_dp():
    dataset_path = './datasets/cifar10/clean-dp/test/'
    model = M.get_RGB(num_classes=2)
    model_path = './models/RGB_dp.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)

def testSRM_fgsm():
    dataset_path = './datasets/cifar10/clean-fgsm/test/'
    model = M.get_SRMModel(num_classes=2)
    model_path = './models/SRM_fgsm.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)

def testSRM_cw():
    dataset_path = './datasets/cifar10/clean-cw/test/'
    model = M.get_SRMModel(num_classes=2)
    model_path = './models/SRM_cw.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)
def testSRM_dp():
    dataset_path = './datasets/cifar10/clean-dp/test/'
    model = M.get_SRMModel(num_classes=2)
    model_path = './models/SRM_dp.pt'
    batch_size= 64
    mini_batchsize = 32
    testFnetModel(dataset_path,model,model_path,batch_size,mini_batchsize)

if __name__=='__main__':
    #===================================
    #testCifar()
    #testFmnist()

    #===================================
    #===================================
    print('===============testFnet_fgsm=========================')
    testFnet_fgsm()
    print('===============testFnet_cw=========================')
    testFnet_cw()
    print('===============testFnet_dp=========================')
    testFnet_dp()

    #===================================
    #===================================
    print('===============testRGB_fgsm=========================')
    testRGB_fgsm()
    print('===============testRGB_cw=========================')
    testRGB_cw()
    print('===============testRGB_dp=========================')
    testRGB_dp()

    #===================================
    #===================================
    print('===============testSRM_fgsm=========================')
    testSRM_fgsm()
    print('===============testSRM_cw=========================')
    testSRM_cw()
    print('===============testSRM_dp=========================')
    testSRM_dp()

    test_KDBU_model('vgg')
