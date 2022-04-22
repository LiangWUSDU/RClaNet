from generators.propose import generator_test_data
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from networks.propose import my_model
from functions.data_gen import *
from functions.ext.medipylib.medipy.metrics import compute_centroid_distance,dice
import nibabel as nib
from functions.write_excel import write_excel_xls_append
import numpy as np
import time
from networks.model_functions import nn_trf
from functions.patch_concat.patch_weight import get_loss_3D_distance_weight
from functions.patch_concat.patch_concat import generator_patch_fast
from functions.metrics import jacobian_determinant,mutil_accuracy,mutil_asd,mutil_hd,mutil_hd95,mutil_assd
learn_rate = 1e-4
test_moving_image_file = 'train_data/test/moving_image/'  #T1图像
test_moving_image_txt = 'train_data/test/test.txt'
test_moving_mask_file = 'train_data/test/moving_mask/'  #T1图像
test_moving_mask_txt = 'train_data/test/test.txt'
train_excel_dir = 'train_data/test/test.xls'  #年龄，性别CDR等
vol_size = (64,64,64)
mode = 'predict'
model = my_model(vol_size=((64,64,64)),enc_nf = [16,32,32,32,16],
							   dec_nf= [32, 32, 32, 32, 16, 3, 16, 32,128,2], mode='predict',  indexing='ij')
train_file = open('train_data/test/test.txt')
train_strings = train_file.readlines()
predict_wrap_dir = "predicts/propose/wrap/"
predict_flow_dir = "predicts/propose/flow/"
predict_wrap_seg_dir = "predicts/propose/wrap_seg/"
##
dice_dir = "predicts/propose/dice.xls"
TRE_dir = "predicts/propose/TRE.xls"
HD_dir = "predicts/propose/HD.xls"
ASD_dir = "predicts/propose/ASD.xls"
##
if not os.path.exists(predict_wrap_dir):
	os.makedirs(predict_wrap_dir)
if not os.path.exists(predict_flow_dir):
	os.makedirs(predict_flow_dir)
if not os.path.exists(predict_wrap_seg_dir):
	os.makedirs(predict_wrap_seg_dir)
##
train_file = open(test_moving_image_txt)
train_strings = train_file.readlines()
train_label_file = open(test_moving_mask_txt)
train_label_strings = train_label_file.readlines()
##
distance_weight_3D = get_loss_3D_distance_weight((64, 64, 64), mode=0)  ##欧几里得距离
distance_weight_4D = distance_weight_3D[...,np.newaxis]
distance_weight_4D = np.concatenate([distance_weight_4D,distance_weight_4D,distance_weight_4D],axis=-1)
##
model.load_weights('weights/propose/weights.03-0.4452.hdf5',by_name=True)
nn_trf_model = nn_trf(vol_size, indexing='ij', interp_method='nearest')
nn_trf_model_linear = nn_trf(vol_size, indexing='ij', interp_method='linear')
for j in range(0, len(train_strings)):
	st = train_strings[j].strip()  # 文件名
	sl = train_label_strings[j].strip()  # 文件名
	moving_image, moving_mask, fixed_image, fixed_mask, location, affine, hdr = generator_test_data\
		(test_moving_image_file, st, test_moving_mask_file, sl)
	moving_image_patch, moving_mask_patch, fixed_image_patch, LOC, location_patch = generator_patch_fast\
		(moving_image,moving_mask, fixed_image, location)
	mask_reg = np.empty((160, 192, 160))
	mask2_reg = np.empty((160, 192, 160))
	mask_flow = np.empty((160, 192, 160, 3))
	mask2_flow = np.empty((160, 192, 160, 3))
	warp_mask = np.empty((160, 192, 160))
	t1 = time.clock()
	for i in range(len(moving_image_patch)):
		input1 = moving_image_patch[i]
		input2 = fixed_image_patch[i]
		input2 = input2[np.newaxis, ..., np.newaxis]
		input3 = location_patch[i]
		input3 = input3[np.newaxis, ...]
		input4 = moving_mask_patch[i]
		input4 = input4[np.newaxis, ..., np.newaxis]
		pred_temp = model.predict([input1, input2, input3])
		pred_flow = pred_temp[0]
		warp_seg_pred = nn_trf_model_linear.predict([input4, pred_flow])[0, ..., 0]
		pred_reg = nn_trf_model_linear.predict([input1, pred_flow])[0, ..., 0]
		mask_reg[LOC[i][0].start:LOC[i][0].stop,LOC[i][1].start:LOC[i][1].stop,
		LOC[i][2].start:LOC[i][2].stop] += pred_reg*distance_weight_3D
		mask2_reg[LOC[i][0].start:LOC[i][0].stop,LOC[i][1].start:LOC[i][1].stop,
		LOC[i][2].start:LOC[i][2].stop] += distance_weight_3D
		##
		mask_flow[LOC[i][0].start:LOC[i][0].stop,LOC[i][1].start:LOC[i][1].stop,
		LOC[i][2].start:LOC[i][2].stop, :] += pred_flow[0, :, :, :, :]*distance_weight_4D
		mask2_flow[LOC[i][0].start:LOC[i][0].stop,LOC[i][1].start:LOC[i][1].stop,
		LOC[i][2].start:LOC[i][2].stop, :] += distance_weight_4D
		##
		warp_mask[LOC[i][0].start:LOC[i][0].stop,LOC[i][1].start:LOC[i][1].stop,
		LOC[i][2].start:LOC[i][2].stop] += warp_seg_pred*distance_weight_3D
		del input1, input2, input3, input4, pred_temp, pred_flow, pred_reg, warp_seg_pred
	t2 = time.clock()
	t = t2-t1
	predict_reg = mask_reg / mask2_reg
	predict_flow = mask_flow / mask2_flow
	predict_wrap_seg = warp_mask / mask2_reg
	predict_reg_image = np.array(predict_reg, dtype=np.float32)
	predict_flow = np.array(predict_flow, dtype=np.float32)
	predict_wrap_seg = np.array(predict_wrap_seg, dtype=np.float32)
	predict_wrap_seg = np.squeeze(predict_wrap_seg)
	predict_Wrap = np.squeeze(predict_reg_image)
	predict_Seg = np.round(predict_wrap_seg)
	nib.save(nib.Nifti1Image(predict_Wrap, affine, hdr), predict_wrap_dir + st)
	nib.save(nib.Nifti1Image(predict_flow, affine, hdr), predict_flow_dir + st)
	nib.save(nib.Nifti1Image(predict_Seg, affine, hdr), predict_wrap_seg_dir + st)
	#jac = jacobian_determinant(predict_flow)
	#jac_negative = jac[jac < 0]
	#jac_num = jac_negative.shape[0]
	#val = dice(predict_Seg, fixed_mask, labels=[1, 2, 3])
	#val2 = compute_centroid_distance(predict_Seg, fixed_mask, labels=[1, 2, 3])
	#val_mean = np.mean(val)
	#val2_mean = np.mean(val2)
	#DSC = [[val[0], val[1], val[2], val_mean, jac_num, t], ]
	#TRE = [[val2[0], val2[1], val2[2], val2_mean], ]
	#print(DSC)
	#print(TRE)
	#write_excel_xls_append(dice_dir, DSC)
	#write_excel_xls_append(TRE_dir, TRE)
	asd, asd_bk = mutil_assd(predict_Seg, fixed_mask)
	hd, hd_bk = mutil_hd(predict_Seg, fixed_mask)
	print(asd)
	print(hd)
	write_excel_xls_append(ASD_dir, asd)
	write_excel_xls_append(HD_dir, hd)
	del st, moving_image, moving_mask, fixed_image, fixed_mask, location, affine, hdr, \
		moving_image_patch, fixed_image_patch, predict_flow, predict_Seg, predict_Wrap






