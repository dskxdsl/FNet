#======================准备数据集=====================
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

import logging
#创建logger并配置用于提示信息
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='01resnet_GenAdv_log2.txt',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=========下载模型===============
import MyModels as M
path = './models/CifarResNet.pt'
net = M.get_CifarResNet()
net.load_state_dict(torch.load(path))
net = net.to(device)
net.eval()
#转换特征
#torchvision.transforms.Compose()是用来设置预处理的方法，其参数是一个list，list中是预处理方法。
path = './cifar-10-norm.txt'
cifar_file = open(path,'rb')
mean = pickle.load(cifar_file)
std = pickle.load(cifar_file)
preprocess = transforms.Compose([
                #transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])

 #获取数据
data_dir = '/Datasets/cifar-10/'
batch_size = 1
_,_,Clean_data_iter,Test_Clean_data_iter = d2l.load_cifar10_dataset(data_dir,batch_size=batch_size,transform=preprocess)

#------------------------------------------------------------------------
#========================================================================


def save_tensor_img(x,path,filename):
    x = x.squeeze(0)
    x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        torch.FloatTensor(mean).view(3, 1, 1)).cpu().detach().numpy()  # reverse of normalization op- "unnormalize"
    x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)

    transform = transforms.ToTensor()
    x_Tensor = transform(x)
    filename = filename + ".png"
    save_dir = os.path.join(path, filename)
    print(save_dir)
    save_image(x_Tensor, save_dir)

def save_adv(method,x_adv,path,true_label,adv_label,filename):
    #print(x_adv.shape)
    x_adv = x_adv.squeeze(0)
    x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
    #print(x_adv.shape)
    x_adv = (x_adv*std)+mean
    x_adv = np.clip(x_adv, 0, 1)

    transform = transforms.ToTensor()
    x_Tensor = transform(x_adv)
    filename = method + '-' + str(true_label)+ '-' + str(adv_label) + '-' + filename  + ".png"
    #print(filename)
    save_dir = os.path.join(path,filename)
    print("save adv file to: ",save_dir)
    save_image(x_Tensor, save_dir)


#============FGSM对抗攻击====================================
from art.attacks.evasion import (FastGradientMethod, BasicIterativeMethod,
                                 DeepFool,CarliniL2Method,CarliniLInfMethod)
from art.classifiers import PyTorchClassifier

#构造art模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001,betas=(0.9,0.99),eps=1e-8,weight_decay=0)

classifier = PyTorchClassifier(model=net,
                               loss=criterion,
                               optimizer=optimizer,
                               input_shape=(1,3,32,32),
                               nb_classes=10
                               )
#============攻击流程===================
def attack_result(Clean_data_iter,method,attack,adv_path):
    logger.info("========================开始{}攻击===============".format(method))
    start = time.time()
    num_corect_clean = 0
    num_corect_adv = 0
    total_num = 0
    for i, item in enumerate(Clean_data_iter):
        logger.info('---------------处理第{}张图片--------------------'.format(i))
        start_gen = time.time()
        data, label = item
        data.to(device)
        label.to(device)
        # ===========处理正确的标签=================
        y_true = label
        target = y_true.detach().numpy()
        logger.info("true label:{}".format(target) )
        # print(type(target))
        # ===========预测干净样本=================
        clean_predictions = classifier.predict(data, top=1)
        clean_label_idx = np.argmax(clean_predictions, axis=1)
        logger.info("clean result:{}".format(clean_label_idx) )
        if target == clean_label_idx:
            #统计正确预测的个数
            num_corect_clean = num_corect_clean + 1
            # ==============制造对抗样本===============
            x_adv = attack.generate(x=data)
            # ===============预测对抗样本==============
            predictions = classifier.predict(x_adv, top=1)
            adv_label_idx = np.argmax(predictions, axis=1)
            logger.info("atack result:{}".format(adv_label_idx) )
            # print(type(adv_label_idx))
            if target == adv_label_idx:
                num_corect_adv = num_corect_adv + 1
                logger.info("attack fail!")
            else:
                logger.info("attack success!")
                logger.info('time {} sec'.format((time.time()-start_gen)) )
                save_adv(method, x_adv, adv_path, target[0], adv_label_idx[0], str(i))
        total_num = i + 1

    num_adv = total_num - num_corect_adv
    accuracy_clean = num_corect_clean / total_num
    accuracy_adv = num_corect_adv / total_num
    atack_success_rate = num_adv / total_num
    logger.info('Totaltime {} sec' .format((time.time() - start)))
    logger.info("共成功制造{}个对抗样本！{}方法的攻击成功率为：{}%".format(num_adv,method, atack_success_rate * 100))
    logger.info("Accuracy on clean examples: {}%".format(accuracy_clean * 100))
    logger.info("Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))
    logger.info("=====================Done!============================")
    logger.info("----------------------------------------")
    logger.info("----------------------------------------")
    logger.info("========================================")

##====================创建文件夹===========================
root_dir = './ResNet_Adv_data/'
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)
adv_FGSM_path = './ResNet_Adv_data/FGSM/'
if not os.path.isdir(adv_FGSM_path):
    os.mkdir(adv_FGSM_path)
adv_BIM_path = './ResNet_Adv_data/BIM/'
if not os.path.isdir(adv_BIM_path):
    os.mkdir(adv_BIM_path)
adv_DP_path = './ResNet_Adv_data/DP/'
if not os.path.isdir(adv_DP_path):
    os.mkdir(adv_DP_path)
adv_CW2_path = './ResNet_Adv_data/CW2/'
if not os.path.isdir(adv_CW2_path):
    os.mkdir(adv_CW2_path)
# adv_CWI_path = './Adv_data/CWI/'
# if not os.path.isdir(adv_CWI_path):
#     os.mkdir(adv_CWI_path)

test_root_dir = './Test_Res_Adv_data/'
if not os.path.isdir(test_root_dir):
    os.mkdir(test_root_dir)
test_adv_FGSM_path = './Test_Res_Adv_data/FGSM/'
if not os.path.isdir(test_adv_FGSM_path):
    os.mkdir(test_adv_FGSM_path)
# test_adv_BIM_path = './Test_Res_Adv_data/BIM/'
# if not os.path.isdir(test_adv_BIM_path):
#     os.mkdir(test_adv_BIM_path)
test_adv_DP_path = './Test_Res_Adv_data/DP/'
if not os.path.isdir(test_adv_DP_path):
    os.mkdir(test_adv_DP_path)
test_adv_CW2_path = './Test_Res_Adv_data/CW2/'
if not os.path.isdir(test_adv_CW2_path):
    os.mkdir(test_adv_CW2_path)
# test_adv_CWI_path = './Test_Adv_data/CWI/'
# if not os.path.isdir(test_adv_CWI_path):
#     os.mkdir(test_adv_CWI_path)

#======================初始化攻击方法=======================
# FGSM_attack = FastGradientMethod(estimator=classifier,norm=np.inf,eps=0.02,eps_step=0.01,batch_size=1,minimal=True)
# BIM_attack = BasicIterativeMethod(estimator=classifier,eps=0.02,eps_step=0.01,batch_size=1)
# DeepFool_attack = DeepFool(classifier,max_iter=100,epsilon=0.02,batch_size=1,verbose=True)
# CarliniL2Method_attack = CarliniL2Method(classifier,targeted=False,max_iter=10,batch_size=1,verbose=True)
#CarliniLInfMethod_attack = CarliniLInfMethod(classifier,targeted=False,max_iter=10,eps=0.02,batch_size=1,verbose=True)

FGSM_attack = FastGradientMethod(estimator=classifier,norm=2,eps=2.0,eps_step=0.1,batch_size=1,minimal=True)
#BIM_attack = BasicIterativeMethod(estimator=classifier,eps=0.1,eps_step=0.01,batch_size=1)
DeepFool_attack = DeepFool(classifier,max_iter=100,epsilon=0.1,batch_size=1,verbose=True)
CarliniL2Method_attack = CarliniL2Method(classifier,targeted=False,learning_rate=0.2,confidence=0.1,max_iter=10,batch_size=1,verbose=True)
#开始攻击
#========================================================================
#----------------------------------------------------------/--------------
#训练集数据
#attack_result(Clean_data_iter,"FGSM",FGSM_attack,adv_FGSM_path)
#attack_result(Clean_data_iter,"BIM",BIM_attack,adv_BIM_path)
#attack_result(Clean_data_iter,"DeepFool",DeepFool_attack,adv_DP_path)
attack_result(Clean_data_iter,"CarliniL2Method",CarliniL2Method_attack,adv_CW2_path)
#attack_result(Clean_data_iter,"CarliniLInfMethod",CarliniLInfMethod_attack,adv_CWI_path)

#测试集数据
attack_result(Test_Clean_data_iter,"FGSM",FGSM_attack,test_adv_FGSM_path)
#attack_result(Test_Clean_data_iter,"BIM",BIM_attack,test_adv_BIM_path)
attack_result(Test_Clean_data_iter,"DeepFool",DeepFool_attack,test_adv_DP_path)
attack_result(Test_Clean_data_iter,"CarliniL2Method",CarliniL2Method_attack,test_adv_CW2_path)
# attack_result(Test_Clean_data_iter,"CarliniLInfMethod",CarliniLInfMethod_attack,test_adv_CWI_path)
#------------------------------------------------------------------------
#========================================================================

import mail as sendmail
msg = "对抗样本已成功生成！"
ret = sendmail.mail(msg=msg)
if ret:
    print("done")
else:
    print("false")


