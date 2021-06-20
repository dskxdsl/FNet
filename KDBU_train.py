import numpy as np
import torch
import torchvision
import pickle
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
import tool as d2l
from KDBU_utils import *
import MyModels as M

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trainKDBU(kdes,lrmodel_path,adv_train_path):
    net = M.get_CifarVGG16()
    net_path = './models/CifarVGG16.pt'
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)
    net.eval()

    norm_path = './cifar-10-norm.txt'
    norm_file = open(norm_path,'rb')
    mean = pickle.load(norm_file)
    std = pickle.load(norm_file)
    preprocess = transforms.Compose([
                    #transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ])
    #======================================================
    #获取数据
    dataset_dir = './Datasets-KDBU/cifar-10/'
    train_set,_,_,_ = d2l.load_cifar10_dataset(dataset_dir=dataset_dir,transform=preprocess)
    # norm_test_path = './adv-vs-clean-vgg/test/clean/'
    # norm_test_set = ImageFolder(norm_test_path,preprocess)
    norm_train_set = train_set
    #adv_train_path = './Datasets-KDBU/adv/train/vgg/fgsm/'
    adv_train_set = ImageFolder(adv_train_path,preprocess)
    #======================================================
    #Bayesian uncertainty
    uncerts_norm_train = get_mc_predictions(net,norm_train_set,50,device).var(axis=0).mean(axis=1)
    uncerts_adv_train = get_mc_predictions(net,adv_train_set,50,device).var(axis=0).mean(axis=1)
    print('train norm BU',uncerts_norm_train.mean())
    print('train adv BU',uncerts_adv_train.mean())
    #======================================================
    #Kernel Density estimation score
    X_norm_train_fratures = get_deep_representations(net,norm_train_set,device)
    X_adv_train_fratures = get_deep_representations(net,adv_train_set,device)
    print('train norm KD',X_norm_train_fratures.mean())
    print('train adv KD',X_adv_train_fratures.mean())

    #======================================================
    #Kernel Density estimation score
    #X_norm_test_fratures = get_deep_representations(net,norm_test_set,device)
    print('Training KDEs...')
    kdes = kdes
    print('Training finished!')
    #======================================================
    # Get model predictions
    print('Computing model predictions...')
    preds_train_normal = X_norm_train_fratures.argmax(axis=1)
    preds_train_adv = X_adv_train_fratures.argmax(axis=1)
    print('Computing prediction finished!')
    densities_normal_train = score_samples(kdes,X_norm_train_fratures,preds_train_normal)
    densities_adv_train = score_samples(kdes,X_adv_train_fratures,preds_train_adv)
    print('Computing densities finished!')

    #======================================================
    uncerts_normal_z_train, uncerts_adv_z_train = Normalize(uncerts_norm_train,uncerts_adv_train)
    densities_normal_z_train, densities_adv_z_train = Normalize(densities_normal_train,densities_adv_train)
    #======================================================
    #train lr
    print('Training start...')
    values, labels, lr = train_lr(
        densities_pos=densities_adv_z_train,
        densities_neg=densities_normal_z_train,
        uncerts_pos=uncerts_adv_z_train,
        uncerts_neg=uncerts_normal_z_train,
        lrmodel_path=lrmodel_path
    )
    print('Training end!')

def train_kdes(base_model,base_model_path):
    net = base_model
    net_path = base_model_path
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)
    net.eval()

    norm_path = './cifar-10-norm.txt'
    norm_file = open(norm_path,'rb')
    mean = pickle.load(norm_file)
    std = pickle.load(norm_file)
    preprocess = transforms.Compose([
                    #transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ])
    #======================================================
    #获取数据
    dataset_dir = './Datasets-KDBU/cifar-10/'
    train_set,_,_,_ = d2l.load_cifar10_dataset(dataset_dir=dataset_dir,transform=preprocess)

    norm_train_set = train_set
    #======================================================
    #======================================================
    #Kernel Density estimation score
    X_norm_train_fratures = get_deep_representations(net,norm_train_set,device)
    print('train norm KD',X_norm_train_fratures.mean())

    #======================================================
    #Kernel Density estimation score
    #X_norm_test_fratures = get_deep_representations(net,norm_test_set,device)
    print('Training KDEs...')
    class_inds = {}
    X_train,y_train = load_keras_dataset(norm_train_set)
    print(X_train.shape)
    print(y_train.shape)
    for i in range(y_train.shape[1]):
        class_inds[i] = np.where(y_train.argmax(axis=1) == i)[0]
    #print(class_inds)
    kdes = {}
    for i in range(y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',bandwidth=0.26).fit(X_norm_train_fratures[class_inds[i]])
    print('Training finished!')
    return kdes

def testKDBU(base_model,base_model_path,kdes,lrmodel_path,adv_test_path):
    net = base_model
    net_path = base_model_path
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)
    net.eval()

    lr = joblib.load(lrmodel_path)

    norm_path = './cifar-10-norm.txt'
    norm_file = open(norm_path,'rb')
    mean = pickle.load(norm_file)
    std = pickle.load(norm_file)
    preprocess = transforms.Compose([
                    #transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ])
    #======================================================
    #获取数据
    dataset_dir = './Datasets-KDBU/cifar-10/'
    train_set,test_set,_,_ = d2l.load_cifar10_dataset(dataset_dir=dataset_dir,transform=preprocess)

    norm_train_set = train_set
    norm_test_set = test_set
    #adv_test_path = './Datasets-KDBU/adv/test/vgg/fgsm/'
    adv_test_set = ImageFolder(adv_test_path,preprocess)
    #======================================================
    #Bayesian uncertainty
    uncerts_norm_test = get_mc_predictions(net,norm_test_set,50,device).var(axis=0).mean(axis=1)
    uncerts_adv_test = get_mc_predictions(net,adv_test_set,50,device).var(axis=0).mean(axis=1)
    print('test norm BU',uncerts_norm_test.mean())
    print('test adv BU',uncerts_adv_test.mean())
    #======================================================
    #Kernel Density estimation score
    #X_norm_train_fratures = get_deep_representations(net,norm_train_set,device)
    X_norm_test_fratures = get_deep_representations(net,norm_test_set,device)
    X_adv_test_fratures = get_deep_representations(net,adv_test_set,device)
    print('test norm KD',X_norm_test_fratures.mean())
    print('test norm KD',X_norm_test_fratures.mean())

    #======================================================
    #Kernel Density estimation score
    #X_norm_test_fratures = get_deep_representations(net,norm_test_set,device)
    print('Training KDEs...')
    kdes = kdes
    print('Training finished!')
    #======================================================
    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = X_norm_test_fratures.argmax(axis=1)
    preds_test_adv = X_adv_test_fratures.argmax(axis=1)
    print('Computing prediction finished!')
    densities_normal_test = score_samples(kdes,X_norm_test_fratures,preds_test_normal)
    densities_adv_test = score_samples(kdes,X_adv_test_fratures,preds_test_adv)
    print('Computing densities finished!')

    #======================================================
    uncerts_normal_z_test, uncerts_adv_z_test = Normalize(uncerts_norm_test,uncerts_adv_test)
    densities_normal_z_test, densities_adv_z_test = Normalize(densities_normal_test,densities_adv_test)

    #======================================================
    values_neg = np.concatenate(
            (densities_normal_z_test.reshape((1, -1)),
            uncerts_normal_z_test.reshape((1, -1))),
            axis=0).transpose([1, 0])
    values_pos = np.concatenate(
            (densities_adv_z_test.reshape((1, -1)),
            uncerts_adv_z_test.reshape((1, -1))),
            axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate((np.zeros_like(densities_normal_z_test), np.ones_like(densities_adv_z_test)))

    probs = lr.predict_proba(values)[:,1]
    print()

    n_samples = len(norm_test_set)
    fpr,tpr,auc_score = compute_roc(probs_neg=probs[:n_samples],probs_pos=probs[n_samples:],labels=labels)
    #print('auc_score:',auc_score)
    return fpr,tpr,auc_score
def get_precision_recall(base_model,base_model_path,kdes,lrmodel_path,adv_test_path):
    net = base_model
    net_path = base_model_path
    net.load_state_dict(torch.load(net_path))
    net = net.to(device)
    net.eval()

    lr = joblib.load(lrmodel_path)

    norm_path = './cifar-10-norm.txt'
    norm_file = open(norm_path,'rb')
    mean = pickle.load(norm_file)
    std = pickle.load(norm_file)
    preprocess = transforms.Compose([
                    #transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ])
    #======================================================
    #获取数据
    dataset_dir = './Datasets-KDBU/cifar-10/'
    train_set,test_set,_,_ = d2l.load_cifar10_dataset(dataset_dir=dataset_dir,transform=preprocess)

    norm_train_set = train_set
    norm_test_set = test_set
    #adv_test_path = './Datasets-KDBU/adv/test/vgg/fgsm/'
    adv_test_set = ImageFolder(adv_test_path,preprocess)
    #======================================================
    #Bayesian uncertainty
    uncerts_norm_test = get_mc_predictions(net,norm_test_set,50,device).var(axis=0).mean(axis=1)
    uncerts_adv_test = get_mc_predictions(net,adv_test_set,50,device).var(axis=0).mean(axis=1)
    print('test norm BU',uncerts_norm_test.mean())
    print('test adv BU',uncerts_adv_test.mean())
    #======================================================
    #Kernel Density estimation score
    #X_norm_train_fratures = get_deep_representations(net,norm_train_set,device)
    X_norm_test_fratures = get_deep_representations(net,norm_test_set,device)
    X_adv_test_fratures = get_deep_representations(net,adv_test_set,device)
    print('test norm KD',X_norm_test_fratures.mean())
    print('test norm KD',X_norm_test_fratures.mean())

    #======================================================
    #Kernel Density estimation score
    #X_norm_test_fratures = get_deep_representations(net,norm_test_set,device)
    print('Training KDEs...')
    kdes = kdes
    print('Training finished!')
    #======================================================
    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = X_norm_test_fratures.argmax(axis=1)
    preds_test_adv = X_adv_test_fratures.argmax(axis=1)
    print('Computing prediction finished!')
    densities_normal_test = score_samples(kdes,X_norm_test_fratures,preds_test_normal)
    densities_adv_test = score_samples(kdes,X_adv_test_fratures,preds_test_adv)
    print('Computing densities finished!')

    #======================================================
    uncerts_normal_z_test, uncerts_adv_z_test = Normalize(uncerts_norm_test,uncerts_adv_test)
    densities_normal_z_test, densities_adv_z_test = Normalize(densities_normal_test,densities_adv_test)

    #======================================================
    values_neg = np.concatenate(
            (densities_normal_z_test.reshape((1, -1)),
            uncerts_normal_z_test.reshape((1, -1))),
            axis=0).transpose([1, 0])
    values_pos = np.concatenate(
            (densities_adv_z_test.reshape((1, -1)),
            uncerts_adv_z_test.reshape((1, -1))),
            axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    y_true = np.concatenate((np.zeros_like(densities_normal_z_test), np.ones_like(densities_adv_z_test)))

    y_pre = lr.predict(values)
    from sklearn.metrics import precision_score,recall_score,accuracy_score

    precision = precision_score(y_true,y_pre,average=None)
    recall = recall_score(y_true,y_pre,average=None)
    accuracy = accuracy_score(y_true,y_pre)
    print('model accuracy:',accuracy)
    print('==========score------norm------adv==============')
    print('precision socre:',precision)
    print('recall socre:',recall)
    print('================================================')
    return precision,recall

def train_all_model():
    base_model = M.get_CifarVGG16()
    base_model_path = './models/CifarVGG16.pt'

    kdes = train_kdes(base_model,base_model_path)

    lrmodel_path = './models/KDBU_fgsm.pkl'
    adv_train_path = './Datasets-KDBU/adv/train/vgg/fgsm/'
    trainKDBU(kdes=kdes,lrmodel_path=lrmodel_path,adv_train_path=adv_train_path)

    lrmodel_path = './models/KDBU_cw.pkl'
    adv_train_path = './Datasets-KDBU/adv/train/vgg/cw/'
    trainKDBU(kdes=kdes,lrmodel_path=lrmodel_path,adv_train_path=adv_train_path)

    lrmodel_path = './models/KDBU_dp.pkl'
    adv_train_path = './Datasets-KDBU/adv/train/vgg/dp/'
    trainKDBU(kdes=kdes,lrmodel_path=lrmodel_path,adv_train_path=adv_train_path)

def val_all_model():
    base_model = M.get_CifarVGG16()
    base_model_path = './models/CifarVGG16.pt'

    kdes = train_kdes(base_model,base_model_path)


    lrmodel_path = './models/KDBU_fgsm.pkl'
    adv_test_path = './Datasets-KDBU/adv/test/vgg/fgsm/'
    testKDBU(base_model,base_model_path,kdes,lrmodel_path,adv_test_path)

    lrmodel_path = './models/KDBU_cw.pkl'
    adv_test_path = './Datasets-KDBU/adv/test/vgg/cw/'
    testKDBU(base_model,base_model_path,kdes,lrmodel_path,adv_test_path)

    lrmodel_path = './models/KDBU_dp.pkl'
    adv_test_path = './Datasets-KDBU/adv/test/vgg/dp/'
    testKDBU(base_model,base_model_path,kdes,lrmodel_path,adv_test_path)

def test_KDBU_model(adv_dataset_name):
    base_model = M.get_CifarVGG16()
    base_model_path = './models/CifarVGG16.pt'

    kdes = train_kdes(base_model,base_model_path)

    print('===============KDBU_fgsm=========================')
    lrmodel_path = './models/KDBU_fgsm.pkl'
    adv_test_path = str('./Datasets-KDBU/adv/test/')+adv_dataset_name+str('/fgsm/')
    get_precision_recall(base_model,base_model_path,kdes,lrmodel_path,adv_test_path)
    print('===============KDBU_cw=========================')
    lrmodel_path = './models/KDBU_cw.pkl'
    adv_test_path = str('./Datasets-KDBU/adv/test/')+adv_dataset_name+str('/cw/')
    get_precision_recall(base_model,base_model_path,kdes,lrmodel_path,adv_test_path)
    print('===============KDBU_dp=========================')
    lrmodel_path = './models/KDBU_dp.pkl'
    adv_test_path = str('./Datasets-KDBU/adv/test/')+adv_dataset_name+str('/dp/')
    get_precision_recall(base_model,base_model_path,kdes,lrmodel_path,adv_test_path)
#tmp()
#train_all_model()
#val_all_model()
