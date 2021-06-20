import torch
import torchvision
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms,models
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pickle
from torch.autograd import Variable
import os.path as path
from sklearn.metrics import roc_curve,auc

import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prosess = transforms.Compose([
    transforms.ToTensor()
])

def mean_max_normlization_np(np_data):
    _range = np_data.max() - np_data.min()
    return (np_data-np_data.min())/_range
def mean_max_normlization_tensor(tensor_data):
    _range = tensor_data.max() - tensor_data.min()
    return (tensor_data-tensor_data.min())/_range

def get_saliency_map(X,y,net):
    net.eval()
    X_var = Variable(X,requires_grad=True)
    y_var = Variable(y)
    saliency = None
    scores = net(X_var)
    scores = scores.gather(1,y_var.view(-1,1)).squeeze()
    #print(scores)
    scores.backward(torch.ones_like(scores))
    saliency = abs(X_var.grad.data)
    saliency,i = torch.max(saliency,dim=1)
    saliency = saliency.squeeze()
    return saliency

def get_mean_std(dataset,mean_std_path):
    means = [0,0,0]
    std = [0,0,0]
    num_imgs = len(dataset)
    for img,y in tqdm(dataset):
        for i in range(3):
            means[i] += img[i,:,:].mean()
            std[i] += img[i,:,:].std()

    means = np.asarray(means)/num_imgs
    std = np.asarray(std)/num_imgs

    # # 将得到的均值和标准差写到文件中，之后就能够从中读取
    with open(mean_std_path, 'wb') as f:
        pickle.dump(means, f)
        pickle.dump(std, f)
        print('pickle done')
    return means, std

def get_2C_mean_std(dataset,mean_std_path):
    means = 0
    std = 0
    num_imgs = len(dataset)
    for img,y in tqdm(dataset):
        means += img.mean()
        std += img.std()

    means = np.asarray(means)/num_imgs
    std = np.asarray(std)/num_imgs

    # # 将得到的均值和标准差写到文件中，之后就能够从中读取
    with open(mean_std_path, 'wb') as f:
        pickle.dump(means, f)
        pickle.dump(std, f)
        print('pickle done')
    return means, std

def load_fmnist_dataset(dataset_dir='/Datasets/FsahionMNIST/',batch_size=128,transform=prosess):
    train_set = torchvision.datasets.FashionMNIST(root=dataset_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root=dataset_dir, train=False, download=True, transform=transform)
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_set, test_set, train_iter, test_iter

def load_cifar10_dataset(dataset_dir='/Datasets/cifar-10/',batch_size=128,transform=prosess):
    train_set = torchvision.datasets.CIFAR10(root=dataset_dir,train=True,download=True,transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=dataset_dir,train=False,download=True,transform=transform)
    train_iter = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=True)
    test_iter = DataLoader(test_set,batch_size=batch_size,shuffle=True,drop_last=True)
    return  train_set,test_set,train_iter,test_iter

def load_local_dataset(dataset_dir,batch_size=128,transform=prosess):
    #获取数据集
    #dataset = ImageFolder(dataset_dir)
    trainset_dir = path.join(dataset_dir,'train/')
    testtrainset_dir = path.join(dataset_dir,'test/')
    train_dataset = ImageFolder(trainset_dir, transform=transform)
    test_dataset = ImageFolder(testtrainset_dir, transform=transform)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    return train_dataset,test_dataset,train_iter,test_iter

def load_trainval_dataset(dataset_dir, ratio = 0.8, batch_size = 2,transform = prosess,num_worker=0):
    #获取数据集
    #dataset = ImageFolder(dataset_dir)
    all_datasets = ImageFolder(dataset_dir, transform=transform)
    print("数据集大小",len(all_datasets))
    #将数据集划分成训练集和测试集
    train_size=int(ratio * len(all_datasets))
    print("训练集大小",train_size)
    test_size=len(all_datasets) - train_size
    print("测试集大小", test_size)
    train_datasets, test_datasets = torch.utils.data.random_split(all_datasets, [train_size, test_size])

    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True,num_workers=num_worker,drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True,num_workers=num_worker,drop_last=True)

    return all_datasets,train_iter,test_iter

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()

def show_bifig(num_epochs,train_data,test_data,x_axis_text,y_axis_text,labeltext1,label_text2,save_path):
    plt.figure()
    plt.plot(np.linspace(0, num_epochs, len(train_data)), train_data, c='r', label=labeltext1)
    plt.plot(np.linspace(0, num_epochs, len(test_data)), test_data, c='b', label=label_text2)
    plt.legend()
    plt.xlabel(x_axis_text)
    plt.ylabel(y_axis_text)
    plt.savefig(save_path)
    # plt.show()
    plt.close()
def show_lossfig(lr,test_data,x_axis_text,y_axis_text,label_text2,save_path):
    plt.figure()
    plt.plot(lr, test_data, c='b', label=label_text2)
    plt.legend()
    plt.xlabel(x_axis_text)
    plt.ylabel(y_axis_text)
    plt.savefig(save_path)
    # plt.show()
    plt.close()

def apply(img, aug, num_row=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_row*num_cols)]
    show_images(Y, num_row, num_cols, scale)

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig('./results/ConfusionMatrix.png')
        #plt.show()
        plt.close()

def eval_model(net,dataset,data_iter,loss,device):
    valid_loss = 0.0
    confusion = ConfusionMatrix(num_classes=len(dataset.classes), labels=dataset.classes)
    with torch.no_grad():
        net.eval()  # 验证
        for inputs, labels in tqdm(data_iter):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = net(inputs)  # 分类网络的输出，分类器用的softmax,即使不使用softmax也不影响分类结果。
            l = loss(output, labels)
            valid_loss += l.item() * inputs.size(0)
            ret, predictions = torch.max(output.data,
                                         1)  # torch.max获取output最大值以及下标，predictions即为预测值（概率最大），这里是获取验证集每个batchsize的预测结果
            # confusion_matrix
            confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())
        confusion.plot()
        confusion.summary()
#================================================
#定义模型
def linreg(X,w,b):
    return torch.mm(X,w) + b
#定义损失函数
def squared_loss(y_hat,y):
    #注意，这里返回的是向量
    return (y_hat - y.view(y_hat.size()))**2/2

#全局平均池化层
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d,self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:])

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)

class ABSLayer(nn.Module):
    def __init__(self):
        super(ABSLayer,self).__init__()
    def forward(self,x):
        return torch.abs(x)

def BilinearPool(y0,y1):
    N = y0.shape[0]
    C0 = y0.shape[1]
    C1 = y1.shape[1]
    feature_size = y0.shape[2] * y0.shape[3]
    y0 = y0.view(N, C0, feature_size)
    y1 = y1.view(N, C1, feature_size)
    y1 = torch.transpose(y1, 1, 2)
    y = (torch.bmm(y0, y1) / feature_size)
    y = y.view(N, -1)
    y = F.normalize(torch.sign(y) * torch.sqrt(torch.abs(y) + 1e-10))
    return y

def show_output_shape(net,data_iter,device):
    net.eval()
    batch = 0
    net = net.to(device)
    for X,y in data_iter:
        X = X.to(device)
        y = y.to(device)
        print('X shape',X.shape)
        y_hat = net(X).cpu()
        print('y_hat',y_hat.shape)
        print('-----------------------')
        batch += 1
        print(batch)
        break

def find_lr(net,trn_loader,batch_size,mini_batch_size,optimizer,criterion,init_value = 1e-8, final_value=10., beta = 0.98,device = 'cuda'):
    import math
    accumulation_steps = batch_size/mini_batch_size
    net = net.to(device)
    net.train()
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    loss_sum = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in tqdm(trn_loader):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        mini_loader = DataLoader(list(zip(*data)),batch_size=mini_batch_size)
        for inputs,labels in mini_loader:
            inputs, labels = inputs.to(device),labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss_sum += loss.data
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.cpu().item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    #return log_lrs, losses
    print(log_lrs)
    plt.plot(log_lrs, losses, c='r', label='lr-loss')
    plt.legend()
    plt.xlabel('lr')
    plt.ylabel('loss')
    plt.savefig('./net_lr_loss.png')
    plt.show()
    plt.close()

def maxnum(a,list):
    max = a
    for item in list:
        max = torch.max(max,item)
    return max

def minnum(a,list):
    min = a
    for item in list:
        min = torch.min(min,item)
    return min


def train_eval(cate, loader, batch_size, mini_batch_size, model, loss_func, optimizer, device):
    preds, labels, loss_sum,acc_sum,acc,n = [], [], 0.,0.,0.,0
    accumulation_steps = batch_size/mini_batch_size
    i = 0
    if cate == "train":
        model.train()
        print('------------train-------------')
        for data in tqdm(loader):
            i += 1
            mini_loader = DataLoader(list(zip(*data)), batch_size=mini_batch_size)
            loss = 0.
            for j, (inputs, targets) in enumerate(mini_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                y = model(inputs)
                loss = loss_func(y, targets)
                loss = loss/accumulation_steps
                preds.append(y.argmax(dim=1))
                labels.append(targets.data)
                loss_sum += loss.cpu().item()
                acc_sum += (y.argmax(dim=1) == targets).sum().cpu().item()
                n += targets.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #loss_sum += loss.data
            #print('train loss batch',i, loss_sum)
        #preds = torch.cat(preds).tolist()
        #labels = torch.cat(labels).tolist()
        loss = loss_sum / len(loader)
        acc = acc_sum / n
    else:
        model.eval()
        print('------------test-------------')
        with torch.no_grad():
            for data in tqdm(loader):
                i += 1
                mini_loader = DataLoader(list(zip(*data)), batch_size=mini_batch_size)
                loss = 0.
                for j, (inputs, targets) in enumerate(mini_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    y = model(inputs)
                    loss = loss_func(y, targets)
                    loss = loss / accumulation_steps
                    preds.append(y.argmax(dim=1))
                    labels.append(targets.data)
                    loss_sum += loss.cpu().item()
                    acc_sum += (y.argmax(dim=1)==targets).sum().cpu().item()
                    n += targets.shape[0]

                #print('test loss batch',i, loss_sum)

            loss = loss_sum / len(loader)
            acc = acc_sum / n

            #print(len(loader))


    return loss, acc, preds, labels


def train_minibatch(batch_size,mini_batch_size, train_iter, test_iter, net, loss_func, optimizer, scheduler, num_epochs, save_path, fig_name,device):
    net = net.to(device)
    min_test_loss = 1
    #max_test_acc = 0.5
    # ============================
    print('training on ', device)
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        print('epoch:',epoch+1)
        #======================================
        #========================================
        train_loss, train_acc, train_pred, train_label = train_eval("train", train_iter,batch_size, mini_batch_size, net, loss_func, optimizer, device)

        test_loss, test_acc, test_pred, test_label = train_eval("eval", test_iter, batch_size, mini_batch_size, net, loss_func, optimizer, device)
        #学习率调整
        scheduler.step()
        # ==============保存最优模型=====================
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            print('epoch %d:已保存最优模型!' % (epoch + 1))
            torch.save(net.state_dict(), save_path)
        # ============================================

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print('epoch %d, train loss %.4f, test loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss, test_loss, train_acc, test_acc, time.time() - start))
        if (epoch + 1) > 2:
            #loss_save_path = './net_loss_epoch' + str(epoch + 1) + '.png'
            loss_file_name = './results/'+fig_name+'net_loss_epoch.png'
            loss_save_path = loss_file_name
            show_bifig((epoch + 1), train_losses, test_losses, 'epoch', 'loss', 'train loss', 'test loss', loss_save_path)

            #acc_save_path = './net_acc_epoch' + str(epoch + 1) + '.png'
            acc_file_name = './results/' + fig_name + 'net_acc_epoch.png'
            acc_save_path = acc_file_name
            show_bifig((epoch + 1), train_accs, test_accs, 'epoch', 'acc', 'train acc', 'test acc', acc_save_path)


def cal_roc(net,test_iter):
    y_list = []
    y_prolist = []
    for X,y in tqdm(test_iter):
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        y_pro = F.softmax(y_hat,dim=1)
        y_pro = y_pro[:,1]
        y_list.append(y.cpu().numpy())
        y_prolist.append(y_pro.detach().cpu().numpy())
    y_list = np.asarray(y_list).reshape(-1,1)
    y_prolist = np.asarray(y_prolist).reshape(-1,1)

    fpr,tpr,thd = roc_curve(y_list,y_prolist)
    AUC = auc(fpr,tpr)
    AUC = ('%.4f' %AUC)
    return fpr,tpr,AUC

def plot_roc(fpr_list,tprlist,auc_list,model_name,pic_title):
    fig = plt.figure()
    legend_list = []
    for i in range(len(model_name)):
        plt.plot(fpr_list[i],tprlist[i])
        legend_list.append(model_name[i]+'(auc:'+str(auc_list[i])+')')
    plt.legend(legend_list)
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Postive Rate')
    plt.title(pic_title)
    filename = str("./results/")+pic_title+str("_ROC.png")
    fig.savefig(filename)
    #plt.show()
    print('Pic Save Success!')
    return
