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
from functions.loss import focal_loss
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from generators.gen_T1 import my_gen_MA
from networks.MultiHeadsModel import mutilheadattModel,MLP
from networks.ResNet import Resnet
from functions.loss import focal_loss
from tensorflow.keras.optimizers import Adam,SGD
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
weight_dir = "new_weights/MA/ResNet/"
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
train_image_file = 'train_data/train1/image/'  #T1图像
train_risk_file = 'train_data/train1/risk/'     # 风险因子
train_txt = 'train_data/train11arg1.txt'
train_excel_dir = 'train_data/train11arg1.xls'  #年龄，性别CDR等
train_name,train_CDR = open_excel_train1(train_excel_dir,'train')
p_C = find_position(train_CDR,0.0)
p_M = find_position(train_CDR,1.0)
p_A = find_position(train_CDR,2.0)
batch_size = 4
##
CN_num = len(p_C)
MCI_num = len(p_M)
AD_num = len(p_A)
epoch = (AD_num + MCI_num)/batch_size
myGene = my_gen_MA(train_risk_file,train_txt,train_CDR,p_M,p_A,batch_size=batch_size)
##
test_ADNI_image_file = 'ADNI/test/test1/image/'
test_ADNI_risk_file = 'ADNI/test/risk1/'
test_txt = 'ADNI/test/test1/test1.txt'
test_excel_dir ='ADNI/test/test1/test1.xls'
test_name, test_CDR = open_excel_train1(test_excel_dir,'test1')
test_CN = 158  # 0
test_MCI = 156 # 1
test_AD = 128   # 2
p_test_C = find_position(test_CDR,0.0)
p_test_M = find_position(test_CDR,1.0)
p_test_A = find_position(test_CDR,2.0)
test_CN_num = len(p_test_C)
test_MCI_num = len(p_test_M)
test_AD_num = len(p_test_A)
test_epoch = (test_MCI_num + test_AD_num)/batch_size
my_validGene = my_gen_MA(test_ADNI_risk_file,test_txt,test_CDR,p_test_M,p_test_A,batch_size=batch_size)

model =  Resnet((160,192,160,1))
#model.summary()
model.compile(optimizer=Adam(lr=0.00001), loss= ['categorical_crossentropy'],metrics=['accuracy'])
model_checkpoint1 = ModelCheckpoint(weight_dir+'weights.{epoch:02d}-{loss:.4f}.hdf5', monitor='loss', verbose=1,
                                    save_best_only=True, save_weights_only=True, mode='auto', period=1)
early_stop = EarlyStopping(monitor='loss', patience=10)
model.fit_generator(myGene,steps_per_epoch = epoch,epochs=100,validation_data=my_validGene,validation_steps=test_epoch,callbacks=[model_checkpoint1,early_stop])

