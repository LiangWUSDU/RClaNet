import tensorflow as tf
import numpy as np

def warp_grid(grid, theta):
    # grid=grid_reference
    num_batch = int(theta.shape[0])
#    theta = tf.cast(tf.reshape(theta, (-1, 3, 4)), 'float32')
    theta = np.reshape(theta, (-1, 3, 4)).astype('float32')
    size = grid.shape
    
    grid = np.concatenate([np.transpose(np.reshape(grid, [-1, 3])), np.ones([1, size[0]*size[1]*size[2]])], axis=0)
    grid = np.reshape(np.tile(np.reshape(grid, [-1]), [num_batch]), [num_batch, 4, -1])
    grid_warped = np.array([np.dot(theta[idx], grid[idx]) for idx in range(num_batch)])
    return np.reshape(np.transpose(grid_warped, [0, 2, 1]), [num_batch, size[0], size[1], size[2], 3])


def resample_linear(inputs, sample_coords):

    input_size = inputs.shape[1:-1]
    spatial_rank = len(inputs.shape) - 2
    samlen = len(sample_coords.shape) - 1
    xy = np.split(sample_coords,sample_coords.shape[-1], axis = samlen)
    xy = [np.squeeze(i, axis = samlen) for i in xy]
    index_voxel_coords = [np.floor(x) for x in xy]

    def boundary_replicate(sample_coords0, input_size0):
        return np.maximum(np.minimum(sample_coords0, input_size0 - 1), 0)

    spatial_coords = [boundary_replicate(x.astype(np.int32), input_size[idx])
                      for idx, x in enumerate(index_voxel_coords)]
    spatial_coords_plus1 = [boundary_replicate(((x+1.).astype(np.int32)), input_size[idx])
                            for idx, x in enumerate(index_voxel_coords)]

    weight = [np.expand_dims(x -np.float32(i), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [np.expand_dims(np.float32(i)- x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = list(spatial_coords[0].shape)
    batch_coords = np.tile(np.reshape(np.arange(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2**spatial_rank)]

#    make_sample = lambda bc: np.gather_nd(inputs, np.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))
#    make_sample = lambda bc: inputs[np.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1)]
#    samples = [make_sample(bc) for bc in binary_codes]
    samples = [make_sample(inputs,batch_coords,sc,bc) for bc in binary_codes]
     
#    samples = make_sample(inputs,batch_coords,sc,binary_codes[0])
        
    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0]*weight_c0[0]+samples0[1]*weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)

def make_sample(inputs,batch_coords,sc,bc):
    x0 = [sc[c][i] for i, c in enumerate(bc)]
    x1 = [batch_coords] + x0
    x2 = np.stack(x1 , -1)
    x3 = inputs[x2[:,:,:,:,0],x2[:,:,:,:,1],x2[:,:,:,:,2],x2[:,:,:,:,3],0]
    x3 = np.reshape(x3,x3.shape + (1,))
#    x3 = tf.gather_nd(inputs,x2)
    return x3


def get_reference_grid(grid_size):
    return np.float32(np.stack(np.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3))


def compute_binary_dice(input1, input2):
    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    vol1 = np.ndarray.sum(np.float32(mask1), axis=[1, 2, 3, 4])
    vol2 = np.ndarray.sum(np.float32(mask2), axis=[1, 2, 3, 4])
    dice = np.ndarray.sum(np.float32(mask1 & mask2), axis=[1, 2, 3, 4])*2 / (vol1+vol2)
    return dice


def compute_centroid_distance(input1, input2, grid=None):
    if grid is None:
        grid = get_reference_grid(input1.shape[1:4])

    def compute_centroid(mask, grid0):
        return np.stack([np.ndarray.sum(np.bool(grid0, mask[i, ..., 0] >= 0.5), axis=0)
                         for i in range(mask.shape[0].value)], axis=0)
    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return np.sqrt(np.ndarray.sum(np.square(c1-c2), axis=1))
