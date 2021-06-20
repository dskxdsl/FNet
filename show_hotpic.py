import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import pickle
import os
import MyModels as M
import tool as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = Image.open('./img/test8904.png')
x_cw = Image.open('./img/CarliniL2Method-0-1-713.png')
x_dp = Image.open('./img/DeepFool-0-1-2233.png')
x_fgsm = Image.open('./img/FGSM-0-1-788.png')

path = './cifar-10-norm.txt'
cifar_file = open(path, 'rb')
mean = pickle.load(cifar_file)
std = pickle.load(cifar_file)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
model_path_list = []
cw_path = './Fnet_cw.pt'
dp_path = './Fnet_dp.pt'
fgsm_path = './Fnet_fgsm.pt'
model_path_list.append(dp_path)
model_path_list.append(cw_path)
model_path_list.append(dp_path)
model_path_list.append(fgsm_path)


x_list = []
x_tensor = transform(x)
x_list.append(x_tensor)
x_cw_tensor = transform(x_cw)
x_list.append(x_cw_tensor)
x_dp_tensor = transform(x_dp)
x_list.append(x_dp_tensor)
x_fgsm_tensor = transform(x_fgsm)
x_list.append(x_fgsm_tensor)

save_list = ['./img/clean.png','./img/cw.png','./img/dp.png','./img/fgsm.png']

#==========================
def save_SM(model_path_list,x_list,save_list):
    net = M.get_FNet(num_classes=2)
    for net_path,x,save_path in zip(model_path_list,x_list,save_list):
        net.load_state_dict(torch.load(net_path))
        net = net.to(device)
        net.eval()
        x = x.unsqueeze(0).to(device)
        y = net(x).argmax(dim=1)
        print(save_path)
        print(y)
        SM = d2l.get_saliency_map(x,y,net).unsqueeze(0)
        #save_image(SM.cpu(),save_path)
        SM_np = SM.cpu().numpy().transpose((1,2,0))
        plt.figure(frameon=False)
        plt.axis('off')
        plt.imshow(SM_np,plt.cm.hot,aspect='equal')
        plt.savefig(save_path)
        print('Done')

def show_dif(adv,x,save_path):
    diff = adv-x
    save_image(diff,save_path)
    print('done')

#===================================
save_SM(model_path_list,x_list,save_list)
plt.figure()
plt.subplot(241)
path = './img/test8904.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('Clean')

plt.subplot(242)
path = './img/FGSM-0-1-788.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('FGSM')

plt.subplot(243)
path = './img/CarliniL2Method-0-1-713.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('CW')

plt.subplot(244)
path = './img/DeepFool-0-1-2233.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('DeepFool')

plt.subplot(245)
path = './img/clean.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('Clean SMap')

plt.subplot(246)
path = './img/fgsm.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('FGSM SMap')

plt.subplot(247)
path = './img/cw.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('CW SMap')

plt.subplot(248)
path = './img/dp.png'
x = plt.imread(path)
plt.axis('off')
plt.imshow(x)
plt.title('Deep Fool SMap')

plt.savefig('./img/result.png')
#plt.show()






