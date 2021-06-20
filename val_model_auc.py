import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import tool as d2l
import MyModels as M
import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def val_fgsm():
    # 预处理标准化
    mean = [ 0.5, 0.5, 0.5 ]
    std  = [ 0.5, 0.5, 0.5 ]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 获取数据
    data_dir = './datasets/cifar10/clean-fgsm/' #'/Datasets/cifar-10/'
    batch_size = 64
    mini_batchsize = 32

    _, test_dataset, _, test_iter = d2l.load_local_dataset(data_dir, batch_size=batch_size, transform=transform)
    #label = dataset.class_to_idx
    #===================================
    # 1.加载模型
    Model_path = './models/Fnet_fgsm.pt'
    net = M.get_FNet(num_classes=2) #
    net.load_state_dict(torch.load(Model_path))
    net = net.to(device)
    net.eval()

    RGBModel_path = './models/RGB_fgsm.pt'
    RGBnet = M.get_RGB(num_classes=2) #
    RGBnet.load_state_dict(torch.load(RGBModel_path))
    RGBnet = RGBnet.to(device)
    RGBnet.eval()

    SRMModel_path = './models/SRM_fgsm.pt'
    SRMnet = M.get_SRMModel(num_classes=2) #
    SRMnet.load_state_dict(torch.load(SRMModel_path))
    SRMnet = SRMnet.to(device)
    SRMnet.eval()
    #===================================

    #=========================================
    ##########################################
    #2. AUC
    fpr_list = []
    tpr_list = []
    AUC_list = []
    model_name = []

    fpr,tpr,AUC = d2l.cal_roc(net,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('Fnet_fgsm')

    fpr,tpr,AUC = d2l.cal_roc(RGBnet,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('RGBnet_fgsm')

    fpr,tpr,AUC = d2l.cal_roc(SRMnet,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('SRMnet_fgsm')

    ##########################################
    #=========================================

    pic_title = 'FGSM(CIFAR_10,VGG16)'
    d2l.plot_roc(fpr_list,tpr_list,AUC_list,model_name,pic_title)

def val_cw():
    # 预处理标准化
    mean = [ 0.5, 0.5, 0.5 ]
    std  = [ 0.5, 0.5, 0.5 ]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 获取数据
    data_dir = './datasets/cifar10/clean-cw/' #'/Datasets/cifar-10/'
    batch_size = 64
    mini_batchsize = 32

    _, test_dataset, _, test_iter = d2l.load_local_dataset(data_dir, batch_size=batch_size, transform=transform)
    #label = dataset.class_to_idx
    #===================================
    # 1.加载模型
    Model_path = './models/Fnet_cw.pt'
    net = M.get_FNet(num_classes=2) #
    net.load_state_dict(torch.load(Model_path))
    net = net.to(device)
    net.eval()

    RGBModel_path = './models/RGB_cw.pt'
    RGBnet = M.get_RGB(num_classes=2) #
    RGBnet.load_state_dict(torch.load(RGBModel_path))
    RGBnet = RGBnet.to(device)
    RGBnet.eval()

    SRMModel_path = './models/SRM_cw.pt'
    SRMnet = M.get_SRMModel(num_classes=2) #
    SRMnet.load_state_dict(torch.load(SRMModel_path))
    SRMnet = SRMnet.to(device)
    SRMnet.eval()
    #=========================================
    ##########################################
    #2. AUC

    fpr_list = []
    tpr_list = []
    AUC_list = []
    model_name = []

    fpr,tpr,AUC = d2l.cal_roc(net,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('Fnet_cw')

    fpr,tpr,AUC = d2l.cal_roc(RGBnet,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('RGBnet_cw')

    fpr,tpr,AUC = d2l.cal_roc(SRMnet,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('SRMnet_cw')
    ##########################################
    #=========================================
    pic_title = 'CW(CIFAR_10,VGG16)'
    d2l.plot_roc(fpr_list,tpr_list,AUC_list,model_name,pic_title)

def val_dp():
    # 预处理标准化
    mean = [ 0.5, 0.5, 0.5 ]
    std  = [ 0.5, 0.5, 0.5 ]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # 获取数据
    data_dir = './datasets/cifar10/clean-dp/' #'/Datasets/cifar-10/'
    batch_size = 64
    mini_batchsize = 32

    _, test_dataset, _, test_iter = d2l.load_local_dataset(data_dir, batch_size=batch_size, transform=transform)
    #label = dataset.class_to_idx
    #===================================
    # 1.加载模型
    Model_path = './models/Fnet_dp.pt'
    net = M.get_FNet(num_classes=2) #
    net.load_state_dict(torch.load(Model_path))
    net = net.to(device)
    net.eval()

    RGBModel_path = './models/RGB_dp.pt'
    RGBnet = M.get_RGB(num_classes=2) #
    RGBnet.load_state_dict(torch.load(RGBModel_path))
    RGBnet = RGBnet.to(device)
    RGBnet.eval()

    SRMModel_path = './models/SRM_dp.pt'
    SRMnet = M.get_SRMModel(num_classes=2) #
    SRMnet.load_state_dict(torch.load(SRMModel_path))
    SRMnet = SRMnet.to(device)
    SRMnet.eval()
    #=========================================
    ##########################################
    #2. AUC

    fpr_list = []
    tpr_list = []
    AUC_list = []
    model_name = []

    fpr,tpr,AUC = d2l.cal_roc(net,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('Fnet_dp')

    fpr,tpr,AUC = d2l.cal_roc(RGBnet,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('RGBnet_dp')

    fpr,tpr,AUC = d2l.cal_roc(SRMnet,test_iter)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    AUC_list.append(AUC)
    model_name.append('SRMnet_dp')
    ##########################################
    #=========================================
    pic_title = 'DeepFool(CIFAR_10,VGG16)'
    d2l.plot_roc(fpr_list,tpr_list,AUC_list,model_name,pic_title)


if __name__ == '__main__':
    #===================================
    val_fgsm()
    val_cw()
    val_dp()
    #===================================
