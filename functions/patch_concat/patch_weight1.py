import numpy as np
import scipy.spatial.distance as dist
import math
def BN(image):
	[x,y,z] = np.shape(image)
	image = np.reshape(image,(x*y*z,1))
	max = np.max(image)
	min = np.min(image)
	image = (image-min)/(max-min)
	image = np.reshape(image,(x,y,z))
	return image
def get_loss_3D_weight(patch_height, patch_width,patch_length, mode, border = 16):
	loss_weight = np.zeros((patch_height, patch_width,patch_height))
	if mode == 0:
		return None
	for i in range(patch_height//2):
		ones = i * np.ones((patch_height - 2 * i, patch_width - 2 * i, patch_length - 2 * i))
		loss_weight[i:patch_height - i, i:patch_width - i, i:patch_length - i] = ones
	max_value = np.max(loss_weight)
	max_value = float(max_value)
	if mode == 4:
		# in this mode, loss weight outside is 0, inner is 1
		loss_weight[np.where(loss_weight < border)] = 0
		loss_weight[np.where(loss_weight >= border)] = 1
		loss_weight = np.reshape(loss_weight, (patch_width * patch_height*patch_length))
	else:
		if mode == 1:
			loss_weight = loss_weight/max_value * loss_weight/max_value
		elif mode == 2:
			loss_weight = loss_weight/max_value
		elif mode == 3:
			loss_weight = np.sqrt(loss_weight/max_value)
	#     loss_weight = np.reshape(loss_weight[:,:,0], (patch_width * patch_height))
	#     loss_weight += 0.01
	#     weight_sum = patch_height * patch_width
	#     cur_sum = np.sum(loss_weight)
	#     loss_weight *= weight_sum/cur_sum

	#loss_weight = np.reshape(loss_weight[:,0], (patch_width*patch_height,1))
	result = BN(loss_weight)
	result = np.reshape(result,(patch_height, patch_width,patch_height,1))
	print("shape of loss_weight:", result.shape)
	return result
def get_loss_3D_distance_weight(winodws, mode,stride):
	patch_height = winodws[0]
	patch_width = winodws[1]
	patch_length = winodws[2]
	loss_weight = np.ones((patch_height, patch_width,patch_length))
	center_x = patch_height / 2 - 1
	center_y = patch_width  / 2 - 1
	center_z = patch_length / 2 - 1
	if mode == 0:  ##欧式距离
		for i in range(patch_height):
			for j in range(patch_width):
				for k in range(patch_length):
					loss_weight[i,j,k] = dist.pdist(np.array([(i,j,k),(center_x,center_y,center_z)]), 'euclidean')
	elif mode == 1: ##曼哈顿距离
		for i in range(patch_height):
			for j in range(patch_width):
				for k in range(patch_length):
					loss_weight[i,j,k] = dist.pdist(np.array([(i,j,k),(center_x,center_y,center_z)]), 'cityblock')
	elif mode == 2: ##堪培拉距离
		for i in range(patch_height):
			for j in range(patch_width):
				for k in range(patch_length):
					loss_weight[i,j,k] = dist.pdist(np.array([(i,j,k),(center_x,center_y,center_z)]), 'canberra')
	elif mode == 3:  ##闵可夫斯基距离
		for i in range(patch_height):
			for j in range(patch_width):
				for k in range(patch_length):
					loss_weight[i, j, k] = dist.pdist(np.vstack([(i, j, k), (center_x, center_y, center_z)]), 'minkowski')
	elif mode == 4:  ##切比雪夫距离
		for i in range(patch_height):
			for j in range(patch_width):
				for k in range(patch_length):
					loss_weight[i, j, k] = dist.pdist(np.vstack([(i, j, k), (center_x, center_y, center_z)]), 'chebyshev')
	elif mode == 5:  ##汉明距离
		for i in range(patch_height):
			for j in range(patch_width):
				for k in range(patch_length):
					loss_weight[i, j, k] = dist.pdist(np.vstack([(i, j, k), (center_x, center_y, center_z)]),
													  'cosine')
	return  loss_weight