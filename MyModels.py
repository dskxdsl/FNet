from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import tool as d2l

class CifarLeNet(nn.Module):
    def __init__(self,class_num = 10):
        super(CifarLeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=84, out_features=class_num)
        #self.maxsoft = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        """
        nn.Linear()的输入输出都是维度为一的值， 所以要把多维度的tensor展平成一维
        x.size()[0] 表示batch_size,  表明映射成 batch_size个 16*5*5 全连接
        """
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.maxsoft(x)
        return x

def get_CifarLeNet():
    net = CifarLeNet()
    return net

class CifarVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarVGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out

def get_CifarVGG16():
    net = CifarVGG16()
    return net

#============================================================
class MnistVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistVGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out

def get_MnistVGG16():
    net = MnistVGG16()
    return net

#=============================================================================

# 定义ResNet基本模块-残差模块
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


# Residual Block
class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = F.relu(out, True)
        return out


class CifarResNet(nn.Module):
    # 实现主module：ResNet34
    # ResNet34 包含多个layer，每个layer又包含多个residual block
    # 用子module来实现residual block，用_make_layer函数来实现layer
    def __init__(self, num_classes=10):
        super(CifarResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(16, 16, 3)
        self.layer2 = self._make_layer(16, 32, 4, stride=1)
        self.layer3 = self._make_layer(32, 64, 6, stride=1)
        self.layer4 = self._make_layer(64, 64, 3, stride=1)
        self.fc = nn.Linear(256, num_classes)  # 分类用的全连接

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建layer,包含多个residual block
        shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 1, stride, bias=False), nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(residual_block(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(residual_block(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_CifarResNet():
    net = CifarResNet()
    return net

#============================================================
# 定义ResNet基本模块-残差模块
class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MnistResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(MnistResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [200,256]
        out = self.outlayer(x)
        return out


def get_MnistResNet():
    net = MnistResNet(Basicblock, [1, 1, 1, 1], 10)
    return net

#===============================================================
#====================================================
#====================================================
#====================================================
#====================================================
#定义和初始化模型
#提取SRM特征
class SRM(nn.Module):
    def __init__(self):
        super(SRM,self).__init__()
        SRM_kernels = [[[[0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., -1., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., -1., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., -1., 1., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., -1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., -1., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., -1., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 1., -1., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., -1., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., -2., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., -2., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 1., -2., 1., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., -2., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., -1., 0., 0.],
                      [0., 0., 3., 0., 0.],
                      [0., 0., -3., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 3., 0.],
                      [0., 0., -3., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 1., -3., 3., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., -3., 0., 0.],
                      [0., 0., 0., 3., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., -3., 0., 0.],
                      [0., 0., 3., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., -3., 0., 0.],
                      [0., 3., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 3., -3., 1., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 3., 0., 0., 0.],
                      [0., 0., -3., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., -1., 2., -1., 0.],
                      [0., 2., -4., 2., 0.],
                      [0., -1., 2., -1., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., -1., 2., -1., 0.],
                      [0., 2., -4., 2., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 2., -1., 0.],
                      [0., 0., -4., 2., 0.],
                      [0., 0., 2., -1., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 2., -4., 2., 0.],
                      [0., -1., 2., -1., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., -1., 2., 0., 0.],
                      [0., 2., -4., 0., 0.],
                      [0., -1., 2., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[-1., 2., -2., 2., -1.],
                      [2., -6., 8., -6., 2.],
                      [-2., 8., -12., 8., -2.],
                      [2., -6., 8., -6., 2.],
                      [-1., 2., -2., 2., -1.]]],

                    [[[-1., 2., -2., 2., -1.],
                      [2., -6., 8., -6., 2.],
                      [-2., 8., -12., 8., -2.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]]],

                    [[[0., 0., -2., 2., -1.],
                      [0., 0., 8., -6., 2.],
                      [0., 0., -12., 8., -2.],
                      [0., 0., 8., -6., 2.],
                      [0., 0., -2., 2., -1.]]],

                    [[[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [-2., 8., -12., 8., -2.],
                      [2., -6., 8., -6., 2.],
                      [-1., 2., -2., 2., -1.]]],

                    [[[-1., 2., -2., 0., 0.],
                      [2., -6., 8., 0., 0.],
                      [-2., 8., -12., 0., 0.],
                      [2., -6., 8., 0., 0.],
                      [-1., 2., -2., 0., 0.]]]]
        SRM_kernels = np.asarray(SRM_kernels,dtype=np.float32)
        SRM_kernels = torch.tensor(SRM_kernels)
        #SRM_kernel = torch.cat((SRM_kernels,SRM_kernels,SRM_kernels),dim=1)
        #print('kernel_size:',SRM_kernels.size()) #(30,1,5,5)
        self.weight = nn.Parameter(data=SRM_kernels,requires_grad=False)


    def forward(self,x):
        #图片的三个通道
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        i = 0

        y_srm = []
        #输出的残差
        #0--7核的残差结果
        inputnum0 = [] #0--7

        inputnum1 = [] #8--11

        inputnum2 = [] #12--19

        inputnum3 = [] #20--24

        inputnum4 = [] #25**29

        for kernel in self.weight:
            kernel = kernel.unsqueeze(0) #kernel (1,1,5,5)
            y1 = F.conv2d(x1,kernel,padding=2)
            y2 = F.conv2d(x2,kernel,padding=2)
            y3 = F.conv2d(x3,kernel,padding=2)

            y = y1+y2+y3
            y_srm.append(y)

            if i <=7:
                inputnum0.append(y)
            elif i <= 11:
                inputnum1.append(y)
            elif i <= 19:
                inputnum2.append(y)
            elif i <= 24:
                inputnum3.append(y)
            else:
                inputnum4.append(y)

            i += 1

        #SRM残差特征
        y_srm = torch.cat(y_srm, dim=1)
        #y_srm = torch.cat((y1_srm,y2_srm,y3_srm),dim=1)

        #print(len(inputnum0_1),len(inputnum1_1),len(inputnum2_1),len(inputnum3_1),len(inputnum4_1))
        #非线性SRM特征
        max0 = d2l.maxnum(inputnum0[0],inputnum0)
        min0 = d2l.minnum(inputnum0[0], inputnum0)

        max1 = d2l.maxnum(inputnum1[0], inputnum1)
        min1 = d2l.minnum(inputnum1[0], inputnum1)

        max2 = d2l.maxnum(inputnum2[0], inputnum2)
        min2 = d2l.minnum(inputnum2[0], inputnum2)

        max3 = d2l.maxnum(inputnum3[0], inputnum3)
        min3 = d2l.minnum(inputnum3[0], inputnum3)

        max4 = d2l.maxnum(inputnum4[0], inputnum4)
        min4 = d2l.minnum(inputnum4[0], inputnum4)

        maxnum1 = torch.cat((max0,max1,max2,max3,max4),dim=1)
        minnum1 = torch.cat((min0,min1,min2,min3,min4),dim=1)
        min_max_srm = torch.cat((maxnum1,minnum1),dim=1)


        #各个通道特征拼接在一起
        total_feature = torch.cat((y_srm,min_max_srm),dim=1)
        #for item in inputnum0_1:
        y = total_feature
        return y

#将噪声特征和图片特征的频率特征拼接在一起，形成提取特征层
class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer,self).__init__()
        self.srm = SRM()
    def forward(self,x):
        y = self.srm(x)
        return y

class FNet(nn.Module):
    def __init__(self,num_classes=2):
        super(FNet,self).__init__()
        self.feature = FeatureLayer()
        self.conv0 = nn.Sequential(
            nn.Conv2d(40, 64, 3),
            d2l.ABSLayer(),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(256**2, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        y0 = self.feature(x) #(N,40,224,224)
        y0 = self.conv0(y0)

        y1 = self.conv1(x)

        y = d2l.BilinearPool(y0,y1)

        y = self.fc(y)

        return y

def get_FNet(num_classes=2):
    net = FNet(num_classes=num_classes)
    # for name,param in net.named_parameters():
    #     if 'weight' in name:
    #         nn.init.normal_(param,mean=0,std=1)
    #     if 'bias' in name:
    #         nn.init.constant_(param,val=0)
    return net
#====================================================
#====================================================
#====================================================
#====================================================
class RGB(nn.Module):
    def __init__(self,num_classes=2):
        super(RGB,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(256*1*1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        y = self.conv1(x)

        y = self.fc(y)

        return y

def get_RGB(num_classes=2):
    net = RGB(num_classes=num_classes)
    # for name,param in net.named_parameters():
    #     if 'weight' in name:
    #         nn.init.normal_(param,mean=0,std=1)
    #     if 'bias' in name:
    #         nn.init.constant_(param,val=0)
    return net

#====================================================
#====================================================
class SRMModel(nn.Module):
    def __init__(self,num_classes=2):
        super(SRMModel,self).__init__()
        self.feature = FeatureLayer()
        self.conv0 = nn.Sequential(
            nn.Conv2d(40, 64, 3),
            d2l.ABSLayer(),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(256*1*1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        y0 = self.feature(x) #(N,40,224,224)
        y = self.conv0(y0)
        y = self.fc(y)

        return y

def get_SRMModel(num_classes=2):
    net = SRMModel(num_classes=num_classes)
    # for name,param in net.named_parameters():
    #     if 'weight' in name:
    #         nn.init.normal_(param,mean=0,std=1)
    #     if 'bias' in name:
    #         nn.init.constant_(param,val=0)
    return net
