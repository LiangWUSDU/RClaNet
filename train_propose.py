# -*- coding: utf-8 -*-
"""
Created on 7.13  10:41:52 2021
@author: LiangWU
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from functions.openexcel import *
from generators.propose import propose_gen
from functions.losses import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from keras.callbacks import ModelCheckpoint,EarlyStopping
from networks.propose import my_model
from keras.optimizers import Adam

weight_dir = "weights/propose/"
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
learn_rate = 1e-4
train_moving_image_file = 'train_data/train/moving_image/'  #T1图像
train_moving_image_txt = 'train_data/train/train.txt'
train_moving_mask_file = 'train_data/train/moving_mask/'  #T1图像
train_moving_mask_txt = 'train_data/train/train.txt'
train_moving_boundary_file = 'train_data/train/moving_mask_boundary/'
train_excel_dir = 'train_data/train/train.xls'  #年龄，性别CDR等
train_mmse,train_age,train_apoe,train_CDR = open_excel(train_excel_dir,750)

myGene = propose_gen(train_moving_image_file,train_moving_image_txt,train_moving_mask_file,train_moving_mask_txt,train_moving_boundary_file,train_CDR,1,1)
model = my_model(vol_size=((64,64,64)),enc_nf = [16,32,32,32,16],dec_nf= [32, 32, 32, 32, 16, 3, 16, 32,128,2], mode='train',  indexing='ij')
model.compile(optimizer=Adam(lr=learn_rate), loss={'flow':reg_loss,'reg_image8':ncchuberloss,'reg_image16':ncchuberloss,'reg_image32':ncchuberloss,'reg_image':ncchuberloss,'coarse_predict':'binary_crossentropy'
    ,'fine_predict':'binary_crossentropy','seg_dice': lambda y_true,y_pred: y_pred,'seg_CSF_dice': lambda y_true,y_pred: y_pred,'seg_GM_dice': lambda y_true,y_pred: y_pred,'seg_WM_dice': lambda y_true,y_pred: y_pred,'seg_boundary_dice':lambda y_true,y_pred: y_pred,
                                                  'seg_CSF_boundary_dice':lambda y_true,y_pred: y_pred,'seg_GM_boundary_dice':lambda y_true,y_pred: y_pred,'seg_WM_boundary_dice':lambda y_true,y_pred: y_pred},loss_weights= [1.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
model_checkpoint1 = ModelCheckpoint(weight_dir+'weights.{epoch:02d}-{loss:.4f}.hdf5', monitor='loss', verbose=1,
                                    save_best_only=True, save_weights_only=True, mode='auto', period=1)
early_stop = EarlyStopping(monitor='loss', patience=5)
model.fit_generator(myGene,steps_per_epoch=60000,epochs=20,callbacks=[model_checkpoint1,early_stop])


