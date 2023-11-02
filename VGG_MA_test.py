# -*- coding: utf-8 -*-
"""
Created on 7.13  10:41:52 2021
@author: LiangWU
"""
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from networks.VGG import VGG16
from functions.write_excel import write_excel_xls_append
from functions.openexcel import *
import numpy as np
import time
import nibabel as nib
from networks.my_model import my_model_single
from functions.metrics import sensitivity,specificity,accuracy
from sklearn.metrics import matthews_corrcoef,accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix
def classification_metric(true,predicts):
	true = np.squeeze(true)
	predicts = np.squeeze(predicts)
	acc = accuracy(true,predicts)
	sen = sensitivity(true,predicts)
	spe = specificity(true,predicts)
	AUC = roc_auc_score(true,predicts)
	MCC = matthews_corrcoef(true,predicts)
	my_confusion_matrix = confusion_matrix(true,predicts)
	metric = [[acc, sen, spe, AUC, MCC], ]
	return metric, my_confusion_matrix
def bn(image):
	[x,y,z] = np.shape(image)
	image = np.reshape(image,(x*y*z,1))
	max = np.max(image)
	min = np.min(image)
	image = (image-min)/(max-min)
	image = np.reshape(image,(x,y,z))
	return image
def find_position(x,value):
	p = []
	for i in range(len(x)):
		if x[i]== value:
			p.append(i)
	p = np.array(p)
	return p
def select_classification_task(moving_strings,y_CDR,p_C,p_A):
	p = np.concatenate([p_C,p_A])
	x = []
	y = []
	for i in range(len(p)):
		x.append(moving_strings[p[i]])
		y.append(y_CDR[p[i]])
	x = np.array(x)
	y = np.array(y)
	return x,y
##OASIS3
OASIS3_image_file = 'OASIS3/test/image/'
OASIS3_risk_file = 'OASIS3/test/risk/'
OASIS3_image_txt = 'OASIS3/test/test.txt'
OASIS3_excel_dir = 'OASIS3/test/test.xls'
OASIS3_file = open(OASIS3_image_txt)
OASIS3_strings = OASIS3_file.readlines()
# CN = 119: 0   MCI = 49 : 1   AD = 17 :2
OASIS3_name,OASIS3_CDR = open_excel_train1(OASIS3_excel_dir,'test')
p_OC = find_position(OASIS3_CDR,0.0)
p_OM = find_position(OASIS3_CDR,1.0)
p_OA = find_position(OASIS3_CDR,2.0)
OASIS3_strings,OASIS3_task = select_classification_task(OASIS3_strings,OASIS3_CDR,p_OM,p_OA)
OASIS3_task[OASIS3_task==1]=0
OASIS3_task[OASIS3_task==2]=1
##AIBL
AIBL_image_file = 'AIBL/test/image/'  #T1图像
AIBL_risk_file = 'AIBL/test/risk/'
AIBL_image_txt = 'AIBL/test/test.txt'
AIBL_excel_dir = 'AIBL/test/test.xls'  #年龄，性别CDR等
AIBL_file = open(AIBL_image_txt)
AIBL_strings = AIBL_file.readlines()
# CN = 110: 0   MCI = 36 : 1   AD = 16 :2
AIBL_name,AIBL_CDR = open_excel_train1(AIBL_excel_dir,'test')
p_AC = find_position(AIBL_CDR,0.0)
p_AM = find_position(AIBL_CDR,1.0)
p_AA = find_position(AIBL_CDR,2.0)
AIBL_strings,AIBL_task = select_classification_task(AIBL_strings,AIBL_CDR,p_AM,p_AA)
AIBL_task[AIBL_task==1]=0
AIBL_task[AIBL_task==2]=1

##ADNI
ADNI_image_file = 'ADNI/test/test1/image/'  #T1图像
ADNI_risk_file = 'ADNI/test/risk1/'
ADNI_image_txt = 'ADNI/test/test1/test1.txt'
ADNI_excel_dir = 'ADNI/test/test1/test1.xls'  #年龄，性别CDR等
ADNI_file = open(ADNI_image_txt)
ADNI_strings = ADNI_file.readlines()
# CN = 188: 0   MCI = 156 : 1   AD = 128 :2
ADNI_name,ADNI_CDR = open_excel_train1(ADNI_excel_dir,'test1')
p_DC = find_position(ADNI_CDR,0.0)
p_DM = find_position(ADNI_CDR,1.0)
p_DA = find_position(ADNI_CDR,2.0)
ADNI_strings,ADNI_task = select_classification_task(ADNI_strings,ADNI_CDR,p_DM,p_DA)
ADNI_task[ADNI_task==1]=0
ADNI_task[ADNI_task==2]=1

## 分类模型
model =  VGG16((160,192,160,1))
weights_path = "new_weights/MA/VGG/"
weights_file = open('new_weights/MA/VGG/weights.txt')
weights_strings = weights_file.readlines()
## 输出文件
out_OASIS3_file = 'new_predicts/VGG/MA/OASIS3.xls'
out_AIBL_file = 'new_predicts/VGG/MA/AIBL.xls'
out_ADNI_file = 'new_predicts/VGG/MA/ADNI.xls'
##
for i in range(0,len(weights_strings)):
	weights = weights_strings[i]
	weights = str.strip(weights)
	model.load_weights(weights_path + weights, by_name=True)
	predicts_OASIS3 = []
	predicts_AIBL = []
	predicts_ADNI = []
	for id in range(len(ADNI_strings)):
		st2 = ADNI_strings[id].strip()
		ADNI_risk = nib.load(ADNI_risk_file + st2).get_data()
		ADNI_risk = bn(ADNI_risk)
		ADNI_risk = ADNI_risk[np.newaxis, ..., np.newaxis]
		pred3 = model.predict( ADNI_risk)
		predict3 = np.argmax(pred3, axis=-1)
		predicts_ADNI.append(predict3)
	predicts_ADNI = np.array(predicts_ADNI)
	ADNI_metric, ADNI_confusion_matrix = classification_metric(ADNI_task,predicts_ADNI)
	print("第"+str(i)+"个权重模型ADNI_metric:",ADNI_metric)
	print(ADNI_confusion_matrix)
	write_excel_xls_append(out_ADNI_file, ADNI_metric)
	for id in range(len(OASIS3_strings)):
		st1 = OASIS3_strings[id].strip()
		OASIS3_risk = nib.load(OASIS3_risk_file + st1).get_data()
		OASIS3_risk = bn(OASIS3_risk)
		OASIS3_risk = OASIS3_risk[np.newaxis, ..., np.newaxis]
		pred1 = model.predict( OASIS3_risk)
		predict1 = np.argmax(pred1, axis=-1)
		predicts_OASIS3.append(predict1)
	predicts_OASIS3 = np.array(predicts_OASIS3)
	OASIS3_metric, OASIS3_confusion_matrix = classification_metric(OASIS3_task,predicts_OASIS3)
	print("第"+str(i)+"个权重模型OASIS3_metric:",OASIS3_metric)
	print(OASIS3_confusion_matrix)
	write_excel_xls_append(out_OASIS3_file, OASIS3_metric)
	for id in range(len(AIBL_strings)):
		st2 = AIBL_strings[id].strip()
		AIBL_risk = nib.load(AIBL_risk_file + st2).get_data()
		AIBL_risk = bn(AIBL_risk)
		AIBL_risk = AIBL_risk[np.newaxis, ..., np.newaxis]
		pred2 = model.predict( AIBL_risk)
		predict2 = np.argmax(pred2, axis=-1)
		predicts_AIBL.append(predict2)
	predicts_AIBL = np.array(predicts_AIBL)
	AIBL_metric, AIBL_confusion_matrix = classification_metric(AIBL_task,predicts_AIBL)
	print("第"+str(i)+"个权重模型AIBL_metric:",AIBL_metric)
	print(AIBL_confusion_matrix)
	write_excel_xls_append(out_AIBL_file, AIBL_metric)



