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
def get_loss_3D_distance_weight(winodws, mode):
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
    loss_weight = np.exp(-np.array(loss_weight))
    return  loss_weight