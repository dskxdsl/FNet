import os
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


sys.stdout = Logger('02_GenDataset_log.txt')

#=======================================================
#====================创建目录============================
#======================================================
import time
start = time.time()
print('-------------start------------------')
#根目录
root_dir = './adv-vs-clean/'
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)
#训练目录
train_dir = os.path.join(root_dir,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
#分类目录
fgsm_dir =os.path.join(train_dir,'01fgsm')
if not os.path.isdir(fgsm_dir):
    os.mkdir(fgsm_dir)
# bim_dir =os.path.join(train_dir,'02bim')
# if not os.path.isdir(bim_dir):
#     os.mkdir(bim_dir)
dp_dir =os.path.join(train_dir,'02dp')
if not os.path.isdir(dp_dir):
    os.mkdir(dp_dir)
cw_dir =os.path.join(train_dir,'03cw')
if not os.path.isdir(cw_dir):
    os.mkdir(cw_dir)
# cwi_dir =os.path.join(train_dir,'05cwi')
# if not os.path.isdir(cwi_dir):
#     os.mkdir(cwi_dir)
clean_dir = os.path.join(train_dir,'00clean')
if not os.path.isdir(clean_dir):
    os.mkdir(clean_dir)

#-----------------------------------------------------
#test目录
test_dir = os.path.join(root_dir,'test')
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)
#分类目录
test_fgsm_dir =os.path.join(test_dir,'01fgsm')
if not os.path.isdir(test_fgsm_dir):
    os.mkdir(test_fgsm_dir)
# test_bim_dir =os.path.join(test_dir,'02bim')
# if not os.path.isdir(test_bim_dir):
#     os.mkdir(test_bim_dir)
test_dp_dir =os.path.join(test_dir,'02dp')
if not os.path.isdir(test_dp_dir):
    os.mkdir(test_dp_dir)
test_cw_dir =os.path.join(test_dir,'03cw')
if not os.path.isdir(test_cw_dir):
    os.mkdir(test_cw_dir)
# test_cwi_dir =os.path.join(test_dir,'05cwi')
# if not os.path.isdir(test_cwi_dir):
#     os.mkdir(test_cwi_dir)
test_clean_dir = os.path.join(test_dir,'00clean')
if not os.path.isdir(test_clean_dir):
    os.mkdir(test_clean_dir)

#====================================================

#复制图片
import shutil
def copy_file(src_dir,dst_dir):
    src_file_list = os.listdir(src_dir)
    num = 0
    for file in src_file_list:
        num += 1
        src_file_dir = os.path.join(src_dir,file)
        dst_file_dir = os.path.join(dst_dir,file)
        print('正在复制第%d个文件：'% num)
        print(src_file_dir)
        print(dst_file_dir)
        shutil.copy(src_file_dir,dst_file_dir)
        print('---------------------------')

#-------------------------------------------------


#=====================================================
#移动adv文件
# adv_BIM_path = './Adv_data/BIM/'
adv_FGSM_path = './Adv_data/FGSM/'
adv_DP_path = './Adv_data/DP/'
adv_CW2_path = './Adv_data/CW2/'
# adv_CWI_path = './Adv_data/CWI/'
copy_file(adv_FGSM_path,fgsm_dir)
# copy_file(adv_BIM_path,bim_dir)
copy_file(adv_DP_path,dp_dir)
copy_file(adv_CW2_path,cw_dir)
# copy_file(adv_CWI_path,cwi_dir)
#---------------------------------------------

#移动clean文件
# clean_dst_dir = clean_dir
# print(clean_dst_dir)
# clean_src_root = './Clean_data/MiniImagenet/'
# clean_src_names = os.listdir(clean_src_root)
# for clean_src_name in clean_src_names:
#     print(clean_src_name)
#     clean_src_dir = os.path.join(clean_src_root,clean_src_name)
#     copy_file(clean_src_dir,clean_dst_dir)
#     print('======================================')
#------------------------------------------------

#=====================================================
#test文件
#移动test adv文件
# test_adv_BIM_path = './Test_Adv_data/BIM/'
test_adv_FGSM_path = './Test_Adv_data/FGSM/'
test_adv_DP_path = './Test_Adv_data/DP/'
test_adv_CW2_path = './Test_Adv_data/CW2/'
# test_adv_CWI_path = './Test_Adv_data/CWI/'
copy_file(test_adv_FGSM_path,test_fgsm_dir)
# copy_file(test_adv_BIM_path,test_bim_dir)
copy_file(test_adv_DP_path,test_dp_dir)
copy_file(test_adv_CW2_path,test_cw_dir)
# copy_file(test_adv_CWI_path,test_cwi_dir)
#---------------------------------------------

#移动移动test clean文件
# test_clean_dst_dir = test_clean_dir
# print(test_clean_dst_dir)
# test_clean_src_root = './Test_Clean_data/MiniImagenet/'
# test_clean_src_names = os.listdir(test_clean_src_root)
# for test_clean_src_name in test_clean_src_names:
#     print(test_clean_src_name)
#     test_clean_src_dir = os.path.join(test_clean_src_root,test_clean_src_name)
#     copy_file(test_clean_src_dir,test_clean_dst_dir)
#     print('======================================')

print('共用时%3f秒'%((time.time()-start)))
print('-------------Done!------------------')