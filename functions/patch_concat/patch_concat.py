import numpy as np
from functions.patch_concat.single_data_gen import vols_generator_patch,vols_location_generator_patch
def softmax(x):
    x_exp = np.exp(-np.array(x))
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s
def daoshu(x):
    # 计算每行的最大值
	x = 1.0/np.array(x)
	x_sum = np.sum(x)
	distance_weight = x / x_sum
	return distance_weight
def generator_patch(moving_image, fixed_image, location):
    moving_image_patch,moving_image_loc = vols_generator_patch(vol_name=moving_image,  patch_size=[64,64,64],
                                                                     stride_patch=[32,32,32], out=2, num_images=80)
    fixed_image_patch = vols_generator_patch(vol_name=fixed_image, patch_size=[64,64,64],
                                             stride_patch=[32,32,32], out=1, num_images=80)
    location_patch = vols_location_generator_patch(vol_name=location,  patch_size=[64,64,64,3],
                                                 stride_patch=[32,32,32,3], out=1, num_images=80)
    moving_image_patch = moving_image_patch[0]
    moving_image_loc = moving_image_loc[0]
    return moving_image_patch,moving_image_loc,fixed_image_patch,location_patch

def generator_patch_fast(moving_image, moving_mask, fixed_image,location, num_image = 80):
    moving_image_patch,moving_image_patch_loc = vols_generator_patch(vol_name=moving_image, patch_size=[64,64,64],
                                                                     stride_patch=[32,32,32], out=2, num_images=num_image)
    fixed_image_patch = vols_generator_patch(vol_name=fixed_image,  patch_size=[64,64,64],
                                             stride_patch=[32,32,32], out=1, num_images=num_image)
    moving_mask_patch = vols_generator_patch(vol_name=moving_mask, patch_size=[64,64,64],
                                             stride_patch=[32,32,32], out=1, num_images=num_image)
    location_patch = vols_location_generator_patch(vol_name=location, patch_size=[64,64,64,3],
                                                   stride_patch=[32,32,32,3], out=1, num_images=num_image)
    moving_image_patch = moving_image_patch[0]
    moving_image_patch_loc = moving_image_patch_loc[0]
    return moving_image_patch,moving_mask_patch,fixed_image_patch,moving_image_patch_loc,location_patch
def weight_distance(flowx,flowy,flowz,weight):
	distance_weight = softmax(weight)
	predict_valuex = np.sum(np.multiply(np.array(flowx), np.array(distance_weight)))
	predict_valuey = np.sum(np.multiply(np.array(flowy), np.array(distance_weight)))
	predict_valuez = np.sum(np.multiply(np.array(flowz), np.array(distance_weight)))
	predict_value = (predict_valuex,predict_valuey,predict_valuez)
	return predict_value
def concat_weight(flowx,flowy,flowz,weight):
	flow = np.empty((160,192,160,3))
	for i in range(160):
		for j in range(192):
			for k in range(160):
				flow[i,j,k,:] = weight_distance(flowx[i][j][k],flowy[i][j][k],flowz[i][j][k],weight[i][j][k])
	return flow
def weight_disk_distance(disk,weight):
	distance_weight = softmax(weight)
	predict_value = np.sum(np.multiply(np.array(disk), np.array(distance_weight)))
	return predict_value
def concat_weight_disk(AD_disk,weight):
	Disk = np.empty((160,192,160))
	for i in range(160):
		for j in range(192):
			for k in range(160):
				Disk[i,j,k] = weight_disk_distance(AD_disk[i][j][k],weight[i][j][k])
	return Disk
def group_list(list1,list2,list3,list4,n): ##n代表每组的个数
	for i in range(0, len(list1), n):
		yield list1[i:i + n],list2[i:i + n],list3[i:i + n],list4[i:i + n]