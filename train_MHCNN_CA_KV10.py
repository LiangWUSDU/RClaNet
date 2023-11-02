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
weight_dir = "new_weights/CA/MHCNN_kf10_10/"
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
##训练1
#p_C_train = p_C[300:]
#p_M_train = p_M[300:]
#p_A_train = p_A[300:]
##验证1
#p_C_valid = p_C[0:300]
#p_M_valid = p_M[0:300]
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
#p_M_train1 = p_M[0:600]
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
#p_C_train1 = p_C[0:900]
#p_C_train2 = p_C[1201:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
#p_M_train1 = p_M[0:900]
#p_M_train2 = p_M[1201:]
#p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
#p_A_train1 = p_A[0:900]
#p_A_train2 = p_A[1201:]
#p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证4
#p_C_valid = p_C[901:1200]
#p_M_valid = p_M[901:1200]
#p_A_valid = p_A[901:1200]
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
##验证6
#p_C_valid = p_C[1501:1800]
#p_M_valid = p_M[1501:1800]
#p_A_valid = p_A[1501:1800]
##
##训练7
#p_C_train1 = p_C[0:1800]
#p_C_train2 = p_C[2101:]
#p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
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
p_C_train1 = p_C[0:2700]
p_C_train2 = p_C[3001:]
p_C_train = np.concatenate((p_C_train1,p_C_train2),axis=-1)
p_M_train1 = p_M[0:2700]
p_M_train2 = p_M[3001:]
p_M_train = np.concatenate((p_M_train1,p_M_train2),axis=-1)
p_A_train1 = p_A[0:2700]
p_A_train2 = p_A[3001:]
p_A_train = np.concatenate((p_A_train1,p_A_train2),axis=-1)
##验证10
p_C_valid = p_C[2701:3000]
p_M_valid = p_M[2701:3000]
p_A_valid = p_A[2701:3000]
##
CN_num = len(p_C_train)
MCI_num = len(p_M_train)
AD_num = len(p_A_train)
epoch = (CN_num + AD_num)/batch_size
valid_epoch = (len(p_C_valid)+len(p_A_valid))/batch_size
myGene = my_gen_kf(train_risk_file,train_txt,train_CDR,p_C_train,p_A_train,batch_size=batch_size)
my_validGene = my_gen_kf(train_risk_file,train_txt,train_CDR,p_C_valid,p_A_valid,batch_size=batch_size)
##
model =  my_model_single_kf((160,192,160,1))
model.summary()
model.load_weights('new_weights/CA/MHCNN_kf10_1/weights.19-0.0295.hdf5',by_name=True)
model.compile(optimizer=Adam(lr=0.00001), loss= ['categorical_crossentropy'],metrics=['accuracy'])
model_checkpoint1 = ModelCheckpoint(weight_dir+'weights.{epoch:02d}-{loss:.4f}.hdf5', monitor='loss', verbose=1,
                                    save_best_only=True, save_weights_only=True, mode='auto', period=1)
early_stop = EarlyStopping(monitor='loss', patience=10)
model.fit_generator(myGene,steps_per_epoch = epoch,epochs=100,validation_data=my_validGene,validation_steps=valid_epoch,callbacks=[model_checkpoint1,early_stop])

