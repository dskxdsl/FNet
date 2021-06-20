import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import pickle
import tool as d2l
import MyModels as M
import mail as sendmail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])
'''
#===================================================================
def train_cifar_model(load_model_param,mean_std_path,dataset_path,model,model_path,model_name):
    cifar_path = mean_std_path
    cifar_file = open(cifar_path, 'rb')
    mean = pickle.load(cifar_file)
    std = pickle.load(cifar_file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #===============
    data_dir = dataset_path #'/Datasets/cifar-10/'
    batch_size = 64
    mini_batchsize = 32

    _, _, train_iter, test_iter = d2l.load_cifar10_dataset(data_dir, batch_size=batch_size, transform=transform)

    '''创建model实例对象，并检测是否支持使用GPU'''
    model = model
    loss_func = nn.CrossEntropyLoss()
    lr = 1e-2
    num_epoch = 60
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    path = model_path

    # d2l.find_lr(model,train_iter,batch_size,mini_batchsize,optimizer,loss_func,lr)
    if load_model_param:
        model.load_state_dict(torch.load(path))
    d2l.train_minibatch(batch_size, mini_batchsize, train_iter, test_iter, model, loss_func, optimizer, scheduler,
                        num_epoch, path, model_name, device)

def train_Fmnist_model(mean_std_path,dataset_path,model,model_path,model_name):
    cifar_path = mean_std_path
    cifar_file = open(cifar_path, 'rb')
    mean = pickle.load(cifar_file)
    std = pickle.load(cifar_file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #===============
    data_dir = dataset_path #'/Datasets/cifar-10/'
    batch_size = 64
    mini_batchsize = 32

    _, _, train_iter, test_iter = d2l.load_fmnist_dataset(data_dir, batch_size=batch_size, transform=transform)

    '''创建model实例对象，并检测是否支持使用GPU'''
    model = model
    loss_func = nn.CrossEntropyLoss()
    lr = 1e-2
    num_epoch = 30
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    path = model_path

    # d2l.find_lr(model,train_iter,batch_size,mini_batchsize,optimizer,loss_func,lr)
    #model.load_state_dict(torch.load(path))
    d2l.train_minibatch(batch_size, mini_batchsize, train_iter, test_iter, model, loss_func, optimizer, scheduler,
                        num_epoch, path, model_name, device)

def trainCifar():
    load_model_param = True
    # train_cifar_model(load_model_param,'./cifar-10-norm.txt', '/Datasets/cifar-10/', M.get_CifarVGG16(), './CifarVGG16.pt','CifarVGG16')
    # train_cifar_model(load_model_param,'./cifar-10-norm.txt','/Datasets/cifar-10/',M.get_CifarResNet(),'./CifarResNet.pt','CifarResNet')
    train_cifar_model(load_model_param,'./cifar-10-norm.txt', '/Datasets/cifar-10/', M.get_CifarLeNet(), './models/CifarLeNet.pt', 'CifarLeNet')


def trainFmnist():
    #train_Fmnist_model('./Fmnist-norm.txt', '/Datasets/FsahionMNIST/', M.get_MnistVGG16(), './FmnistVGG16.pt','FmnistVGG16')
    train_Fmnist_model('./Fmnist-norm.txt','/Datasets/FsahionMNIST/',M.get_MnistResNet(),'./models/FmnistResNet.pt','FmnistResNet')

#===================================================================
#===================================================================

#===================================================================
#===================================================================
def train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler):
    mean = [ 0.5, 0.5, 0.5 ]
    std  = [ 0.5, 0.5, 0.5 ]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #===============
    data_dir = dataset_path #'/Datasets/cifar-10/'
    batch_size = batch_size
    mini_batchsize = mini_batchsize

    #train_set,test_set, train_iter, test_iter = d2l.load_local_dataset(data_dir, batch_size=batch_size, transform=transform)
    train_set,train_iter,test_iter = d2l.load_trainval_dataset(data_dir,batch_size=batch_size, transform=transform)
    print(train_set.classes)
    '''创建model实例对象，并检测是否支持使用GPU'''
    path = model_path

    #d2l.show_output_shape(model,val_iter,device)
    # d2l.find_lr(model,train_iter,batch_size,mini_batchsize,optimizer,loss_func,lr)
    if load_model_param:
        model.load_state_dict(torch.load(path))
    d2l.train_minibatch(batch_size, mini_batchsize, train_iter, test_iter, model, loss_func, optimizer, scheduler,
                        num_epoch, path, model_name, device)

#===================================================================
#===================================================================

def trainFnet_fgsm():
    dataset_path = './datasets/cifar10/clean-fgsm/train/'
    model = M.get_FNet(num_classes=2)
    print(model)
    model_path = './models/Fnet_fgsm.pt'
    model_name = 'Fnet_fgsm'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)
def trainFnet_cw():
    dataset_path = './datasets/cifar10/clean-cw/train/'
    model = M.get_FNet(num_classes=2)
    print(model)
    model_path = './models/Fnet_cw.pt'
    model_name = 'Fnet_cw'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)
def trainFnet_dp():
    dataset_path = './datasets/cifar10/clean-dp/train/'
    model = M.get_FNet(num_classes=2)
    print(model)
    model_path = './models/Fnet_dp.pt'
    model_name = 'Fnet_dp'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)

#===================================================================
#===================================================================
def trainRGB_fgsm():
    dataset_path = './datasets/cifar10/clean-fgsm/train/'
    model = M.get_RGB(num_classes=2)
    print(model)
    model_path = './models/RGB_fgsm.pt'
    model_name = 'RGB_fgsm'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)
def trainRGB_cw():
    dataset_path = './datasets/cifar10/clean-cw/train/'
    model = M.get_RGB(num_classes=2)
    print(model)
    model_path = './models/RGB_cw.pt'
    model_name = 'RGB_cw'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)
def trainRGB_dp():
    dataset_path = './datasets/cifar10/clean-dp/train/'
    model = M.get_RGB(num_classes=2)
    print(model)
    model_path = './models/RGB_dp.pt'
    model_name = 'RGB_dp'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)

def trainSRM_fgsm():
    dataset_path = './datasets/cifar10/clean-fgsm/train/'
    model = M.get_SRMModel(num_classes=2)
    print(model)
    model_path = './models/SRM_fgsm.pt'
    model_name = 'SRM_fgsm'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)
def trainSRM_cw():
    dataset_path = './datasets/cifar10/clean-cw/train/'
    model = M.get_SRMModel(num_classes=2)
    print(model)
    model_path = './models/SRM_cw.pt'
    model_name = 'SRM_cw'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)
def trainSRM_dp():
    dataset_path = './datasets/cifar10/clean-dp/train/'
    model = M.get_SRMModel(num_classes=2)
    print(model)
    model_path = './models/SRM_dp.pt'
    model_name = 'SRM_dp'
    lr=1e-1
    num_epoch = 100
    batch_size= 64
    mini_batchsize = 32
    load_model_param = False

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_FNet_model(load_model_param,dataset_path,model,model_path,model_name,lr,num_epoch,batch_size,mini_batchsize,loss_func,optimizer,scheduler)

def main():
    #===================================
    #trainCifar()
    #trainFmnist()

    #===================================
    trainFnet_fgsm()
    trainFnet_cw()
    trainFnet_dp()
    #===================================
    #===================================
    trainRGB_fgsm()
    trainRGB_cw()
    trainRGB_dp()
    #
    trainSRM_fgsm()
    trainSRM_cw()
    trainSRM_dp()
    #===================================
    #===================================
    msg = 'fnet-cifar done!'
    ret = sendmail.mail(msg=msg)
    if ret:
        print('Done')
    else:
        print('false')


if __name__ == '__main__':
    main()
