import numpy as np
from functions.patch_concat.single_data_gen import vols_generator_patch,vols_location_generator_patch
def softmax(x):
    # 计算每行的最大值
    #row_max = np.max(x)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    #x = x - row_max
    # 计算e的指数次幂
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
def generator_patch(moving_image, fixed_image, stride,num_image):
	moving_image_patch = vols_generator_patch(vol_name=moving_image,  patch_size=[64,64,64],
											  stride_patch=stride, out=1, num_images=num_image)
	fixed_image_patch = vols_generator_patch(vol_name=fixed_image,patch_size=[64,64,64],
											 stride_patch=stride, out=1, num_images=num_image)
	return moving_image_patch,fixed_image_patch
def weight_distance(flow,weight):
	h = len(flow)
	if h==1:
		predict_value = np.array(flow)
	else:
		weight_sum = np.sum(weight)
		distance_weight = weight / weight_sum
		predict_value = np.sum(np.array(flow)*np.array(distance_weight))
	return predict_value
def weight_distance1(flowx,flowy,flowz,weight):
	distance_weight = softmax(weight)
	predict_valuex = np.sum(np.multiply(np.array(flowx), np.array(distance_weight)))
	predict_valuey = np.sum(np.multiply(np.array(flowy), np.array(distance_weight)))
	predict_valuez = np.sum(np.multiply(np.array(flowz), np.array(distance_weight)))
	predict_value = (predict_valuex,predict_valuey,predict_valuez)
	return predict_value
def weight_max_distance(flowx,flowy,flowz,weight):
	h = len(flowx)
	if h <2:
		weight_sum = np.sum(weight)
		distance_weight = weight / weight_sum
		predict_valuex = np.sum(np.array(flowx)*np.array(distance_weight))
		predict_valuey = np.sum(np.array(flowy)*np.array(distance_weight))
		predict_valuez = np.sum(np.array(flowz)*np.array(distance_weight))
		predict_value = (predict_valuex,predict_valuey,predict_valuez)
	else:
		maxx = weight.index(np.max(flowx))
		minx =  weight.index(np.min(flowx))
		maxy = weight.index(np.max(flowy))
		miny =  weight.index(np.min(flowy))
		maxz = weight.index(np.max(flowz))
		minz =  weight.index(np.min(flowz))
		weightx = weighty = weightz = weight
		del flowx[maxx],flowx[minx-1],weightx[maxx],weightx[minx-1],\
			flowy[maxy],flowy[miny-1],weighty[maxy],weighty[miny-1],\
			flowz[maxz],flowz[minz-1],weightz[maxz],weightz[minz-1]
		weight_sumx = np.sum(weightx)
		distance_weightx = weightx / weight_sumx
		weight_sumy = np.sum(weighty)
		distance_weighty = weighty / weight_sumy
		weight_sumz = np.sum(weightz)
		distance_weightz = weightz / weight_sumz
		predict_valuex = np.sum(np.multiply(np.array(flowx),np.array(distance_weightx)))
		predict_valuey = np.sum(np.multiply(np.array(flowy),np.array(distance_weighty)))
		predict_valuez = np.sum(np.multiply(np.array(flowz),np.array(distance_weightz)))
		predict_value = (predict_valuex,predict_valuey,predict_valuez)
	return predict_value

def concat_weight(flowx,flowy,flowz,weight):
	flow = np.empty((160,192,160,3))
	for i in range(160):
		for j in range(192):
			for k in range(160):
				flow[i,j,k,:] = weight_distance1(flowx[i][j][k],flowy[i][j][k],flowz[i][j][k],weight[i][j][k])
	return flow
def concat_seg(flow):
	flow_x = np.ones((160,192,160))
	for i in range(160):
		for j in range(192):
			for k in range(160):
				floww = flow[i][j][k]
				maxlabel = max(floww, key=floww.count)
				flow_x[i,j,k] = maxlabel
	return flow_x
def group_list(list1,list2,list3, n): ##n代表每组的个数
	for i in range(0, len(list1), n):
		yield list1[i:i + n],list2[i:i + n],list3[i:i + n]