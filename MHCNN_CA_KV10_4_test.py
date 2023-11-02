# -*- coding: utf-8 -*-
"""
Created on 7.13  10:41:52 2021
@author: LiangWU
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from functions.openexcel import open_excel_train1
from functions.losses import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from functions.loss import focal_loss
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from generators.gen_T1 import my_gen_kf
from networks.MultiHeadsModel import mutilheadattModel,MLP
from networks.my_model import my_model_single_kf
from functions.loss import focal_loss
from tensorflow.keras.optimizers import Adam
from functions.metrics import sensitivity,specificity,accuracy
import nibabel as nib
from sklearn.metrics import matthews_corrcoef,accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix
# CN vs. AD   CA
# CN vs. MCI  CM
# MCI vs. AD   MA
def find_position(x,value):
    p = []
    for i in range(len(x)):
        if x[i]== value:
            p.append(i)
    p = np.array(p)
    return p
def classification_metric(true,predicts):
	true = np.squeeze(true)
	predicts = np.squeeze(predicts)
	acc = accuracy(true,predicts)
	sen = sensitivity(true,predicts)
	spe = specificity(true,predicts)
	#AUC = roc_auc_score(true,predicts)
	#MCC = matthews_corrcoef(true,predicts)
	f1 = f1_score(true,predicts)
	my_confusion_matrix = confusion_matrix(true,predicts)
	metric = [[acc, sen, spe,f1], ]
	return metric, my_confusion_matrix
def bn(image):
	[x,y,z] = np.shape(image)
	image = np.reshape(image,(x*y*z,1))
	max = np.max(image)
	min = np.min(image)
	image = (image-min)/(max-min)
	image = np.reshape(image,(x,y,z))
	return image
train_image_file = 'train_data/train1/image/'  #T1图像
train_risk_file = 'train_data/train1/risk/'     # 风险因子
train_txt = 'train_data/train11arg1.txt'
train_excel_dir = 'train_data/train11arg1.xls'  #年龄，性别CDR等
train_name,train_CDR = open_excel_train1(train_excel_dir,'train')
p_C = find_position(train_CDR,0.0)
p_M = find_position(train_CDR,1.0)
p_A = find_position(train_CDR,2.0)
batch_size = 4
##训练1
#p_C_train = p_C[300:]
#p_M_train = p_M[300:]
#p_A_train = p_A[300:]
##验证1
#p_C_valid = p_C[0:300]
##p_M_valid = p_M[0:300]
#p_A_valid = p_A[0:300]
##
##训练2
#p_C_train1 = p_C[0:300]
#p_C_train2 = p_C[601:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:300]
#p_M_train2 = p_M[601:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:300]
#p_A_train2 = p_A[601:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
#验证2
#p_C_valid = p_C[301:600]
#p_M_valid = p_M[301:600]
#p_A_valid = p_A[301:600]
##
##训练3
#p_C_train1 = p_C[0:600]
#p_C_train2 = p_C[901:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
##p_M_train1 = p_M[0:600]
#p_M_train2 = p_M[901:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:600]
#p_A_train2 = p_A[901:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证3
#p_C_valid = p_C[601:900]
#p_M_valid = p_M[601:900]
#p_A_valid = p_A[601:900]
##
##训练4
p_C_train1 = p_C[0:900]
p_C_train2 = p_C[1201:]
p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
p_M_train1 = p_M[0:900]
p_M_train2 = p_M[1201:]
p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
p_A_train1 = p_A[0:900]
p_A_train2 = p_A[1201:]
p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证4
p_C_valid = p_C[901:1200]
p_M_valid = p_M[901:1200]
p_A_valid = p_A[901:1200]
##
##训练5
#p_C_train1 = p_C[0:1200]
#p_C_train2 = p_C[1501:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:1200]
#p_M_train2 = p_M[1501:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:1200]
#p_A_train2 = p_A[1501:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证5
#p_C_valid = p_C[1201:1500]
#p_M_valid = p_M[1201:1500]
#p_A_valid = p_A[1201:1500]
##
##训练6
#p_C_train1 = p_C[0:1500]
#p_C_train2 = p_C[1801:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:1500]
#p_M_train2 = p_M[1801:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:1500]
#p_A_train2 = p_A[1801:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证5
#p_C_valid = p_C[1501:1800]
#p_M_valid = p_M[1501:1800]
#p_A_valid = p_A[1501:1800]
##
##训练7
#p_C_train1 = p_C[0:1800]
#p_C_train2 = p_C[2101:]
##p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:1800]
#p_M_train2 = p_M[2101:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:1800]
#p_A_train2 = p_A[2101:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
#验证7
#p_C_valid = p_C[1801:2100]
#p_M_valid = p_M[1801:2100]
#p_A_valid = p_A[1801:2100]
##
##训练8
#p_C_train1 = p_C[0:2100]
#p_C_train2 = p_C[2401:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:2100]
#p_M_train2 = p_M[2401:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:2100]
#p_A_train2 = p_A[2401:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证8
#p_C_valid = p_C[2101:2400]
#p_M_valid = p_M[2101:2400]
#p_A_valid = p_A[2101:2400]
##
##训练9
#p_C_train1 = p_C[0:2400]
#p_C_train2 = p_C[2701:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:2400]
#p_M_train2 = p_M[2701:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:2400]
#p_A_train2 = p_A[2701:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证9
#p_C_valid = p_C[2401:2700]
#p_M_valid = p_M[2401:2700]
#p_A_valid = p_A[2401:2700]
##
##训练10
#p_C_train1 = p_C[0:2700]
#p_C_train2 = p_C[3001:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:2700]
#p_M_train2 = p_M[3001:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:2700]
#p_A_train2 = p_A[3001:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证10
#p_C_valid = p_C[2701:3000]
#p_M_valid = p_M[2701:3000]
#p_A_valid = p_A[2701:3000]
##
##
model =  my_model_single_kf((160,192,160,1))
def Random(moving_strings,y_CDR,p_C,p_A):
    p = np.concatenate([p_C,p_A])
    x = []
    y = []
    for i in range(len(p)):
        x.append(moving_strings[p[i]])
        y.append(y_CDR[p[i]])
    x = np.array(x)
    y = np.array(y)
    return x,y
def generator_data(train_moving_image_file, st):
    vol_dir = train_moving_image_file + st
    image1 = nib.load(vol_dir)
    image = image1.get_fdata()
    affine0 = image1.affine.copy()
    moving_image = np.asarray(image, dtype=np.float32)
    #moving_image = np.resize(moving_image,(64,64,64))
    moving_image = bn(moving_image)
    #moving_image = moving_image[32:128,32:160,32:128]
    return moving_image,affine0
##
weights_path = "new_weights/CA/MHCNN_kf10_4/"
weights_file = open('new_weights/CA/MHCNN_kf10_4/weights.txt')
weights_strings = weights_file.readlines()
##
from functions.write_excel import write_excel_xls_append

for i in range(0,len(weights_strings)):
    weights = weights_strings[i]
    weights = str.strip(weights)
    model.load_weights(weights_path + weights, by_name=True)
    image_file = open(train_txt)
    image_strings = image_file.readlines()
    image_strings,y_CDR = Random(image_strings,train_CDR,p_C_valid,p_A_valid)
    y_CDR[y_CDR==0]=0
    y_CDR[y_CDR==2]=1
    predicts = []
    for id in range(len(image_strings)):
        st = image_strings[id].strip()  # 文件名
        y_label = y_CDR[id]
        moving_image,affine0 = generator_data(train_risk_file, st)
        moving_image = moving_image[np.newaxis, ..., np.newaxis]
        pred3 = model.predict(moving_image)
        predict3 = np.argmax(pred3, axis=-1)
        predicts.append(predict3)
    predicts = np.array(predicts)
    ADNI_metric, ADNI_confusion_matrix = classification_metric(y_CDR, predicts)
    print(ADNI_metric)
    write_excel_xls_append('new_predicts/MHCNN/KV10/CA.xls', ADNI_metric)



