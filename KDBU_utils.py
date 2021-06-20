import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import time
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

import multiprocessing as mp

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision.transforms import transforms

from tqdm import tqdm
import time
import warnings

from sklearn.preprocessing import scale
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegressionCV
import joblib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26}

def Normalize(data,adv):
    data = (data-0.5)/0.5
    adv = (adv-0.5)/0.5
    return data,adv

prosess = transforms.Compose([
    transforms.ToTensor()
])
def load_keras_dataset(data_set,batch_size=1):
    data_iter = DataLoader(data_set,batch_size,shuffle=False,drop_last=False)
    train_X = []
    train_Y = []

    for X,y in data_iter:
        X = X.squeeze(0).numpy()
        train_X.append(X)
        y = y.numpy()
        y = np.eye(10)[y]
        y = y.squeeze(0)
        train_Y.append(y)
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)
    return train_X,train_Y

def flip(x, nb_diff):
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 1.

    return np.reshape(x, original_shape)

def classifier_adv(model,data):
    return F.log_softmax(model(data))

def get_mc_predictions(model, data_set, nb_iter,device):
    model = model.to(device)
    model.eval()
    preds_mc = []
    all_prob = []
    for X,y in tqdm(data_set):
        X = X.to(device)
        #print(X.shape)
        y_hat = model(X.unsqueeze(0))
        prob = F.softmax(y_hat, dim=1).detach().cpu().numpy()
        #print(prob.shape)
        all_prob.append(prob)
    all_prob = np.asarray(all_prob).reshape((-1,10))
    for i in tqdm(range(nb_iter)):
        preds_mc.append(all_prob)
    return np.asarray(preds_mc)

def get_deep_representations(model, data_set, device):
    model = model.to(device)
    model.eval()
    #X = torch.Tensor(X).to(device)
    # mnist modelA last hidden layer
    #     output_dim = model.layers[-4].output.shape[-1].value
    output_dim = list(model.children())[-1][-1].out_features
    #print(output_dim)
    output = []
    for X, y in tqdm(data_set):
        X = X.to(device)
        # print(X.shape)
        y_hat = model(X.unsqueeze(0)).detach().cpu().numpy()
        # print(prob.shape)
        output.append(y_hat)
    output = np.asarray(output).reshape((-1, output_dim))
    return output

def score_samples(kdes, samples, preds):
    results = []
    for x,i in zip(samples,preds):
        x = x.reshape((1,-1))
        kde = kdes[i].score_samples(x)[0]
        results.append(kde)
        #print(kde)

    return np.asarray(results)

def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg,lrmodel_path):

    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)
    joblib.dump(lr,lrmodel_path)
    return values, labels, lr

def compute_roc(probs_neg,probs_pos,labels,plot=False):
    probs = np.concatenate((probs_neg,probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg),np.ones_like(probs_pos)))
    fpr,tpr,_ = roc_curve(labels,probs)
    auc_score = auc(fpr,tpr)
    if plot:
        plt.figure()
        plt.plot(fpr,tpr,label='ROC (AUC=%0.4f)' % auc_score)
        plt.plot([0,1],[0,1])
        plt.legend()
        plt.title('ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig('./results/KDBU_AUC.png')
    return fpr,tpr,auc_score
