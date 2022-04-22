import numpy as np
from functions.binary import *
import functions.pynd.ndutils as nd
import tensorflow as tf
from functions.util import *


def Get_Jac(displacement):
    displacement = displacement[np.newaxis,...]
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3
    return D

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2
        dfdx = J[0]
        dfdy = J[1]
        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
def jac_from_output(output):
    a = []
    output = output[np.newaxis,...]
    for i in output:
        jac = jacobian_determinant(i)
        jac_negative = jac[jac < 0]
        a.append(jac_negative.shape[0])
    return a
def get_reference_grid(grid_size):
    return tf.to_float(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3))


def compute_binary_dice(input1, input2):
    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    vol1 = tf.reduce_sum(tf.to_float(mask1), axis=[1, 2, 3, 4])
    vol2 = tf.reduce_sum(tf.to_float(mask2), axis=[1, 2, 3, 4])
    dice = tf.reduce_sum(tf.to_float(mask1 & mask2), axis=[1, 2, 3, 4])*2 / (vol1+vol2)
    return dice


def compute_centroid_distance(input1, input2, grid=None):
    if grid is None:
        grid = get_reference_grid(input1.get_shape()[1:4])

    def compute_centroid(mask, grid0):
        return tf.stack([tf.reduce_mean(tf.boolean_mask(grid0, mask[i, ..., 0] >= 0.5), axis=0)
                         for i in range(mask.shape[0].value)], axis=0)
    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1-c2), axis=1))

def compute_centroid_distance_val(input1, input2):

    input1 = np.reshape(input1,(1,160,192,160,1))
    input2 = np.reshape(input2,(1,160,192,160,1))
    centroid_distance=compute_centroid_distance(tf.convert_to_tensor(input1, dtype=tf.float32), tf.convert_to_tensor(input2, dtype=tf.float32), grid=None)
    with tf.Session() as sess:
        return sess.run(centroid_distance)
    #return centroid_distance


def multi_class(x):
    CSF = np.zeros((160,192,160))
    GM = np.zeros((160,192,160))
    WM = np.zeros((160,192,160))
    CSF[x==1]=1
    GM[x==2]=1
    WM[x==3]=1
    return CSF,GM,WM
def compute_centroid_distance_val_all(input1,input2):
    CSF1, GM1, WM1 = multi_class(input1)
    CSF2, GM2, WM2 = multi_class(input2)
    tre1 = compute_centroid_distance_val(CSF1,CSF2)
    tre2 = compute_centroid_distance_val(GM1,GM2)
    tre3 = compute_centroid_distance_val(WM1,WM2)
    tre4 = compute_centroid_distance_val(input1,input2)
    return tre1,tre2,tre3,tre4


def mutil_seg(label):
    label = np.reshape(label,(160,192,160))
    BK = np.zeros((160,192,160))
    CSF = np.zeros((160,192,160))
    GM = np.zeros((160,192,160))
    WM = np.zeros((160,192,160))
    BK[label ==0] =1
    CSF[label==1] =1
    GM[label ==2] =1
    WM[label ==3] =1
    return BK,CSF,GM,WM


def accuracy(y_true, y_pred):
    # https://stackoverflow.com/a/27475514
    y_true = np.reshape(y_true,(160*192*160,1))
    y_pred = np.reshape(y_pred,(160*192*160,1))
    true_positive = len(np.where((y_true == 1) & (y_pred == 1))[0])
    true_negative = len(np.where((y_true == 0) & (y_pred == 0))[0])
    false_positive = len(np.where((y_true == 0) & (y_pred == 1))[0])
    false_negative = len(np.where((y_true == 1) & (y_pred == 0))[0])
    acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    return acc
def mutil_accuracy(y_true, y_pred):
    T_BK,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = accuracy(T_CSF,P_CSF)
    a2 = accuracy(T_GM,P_GM)
    a3 = accuracy(T_WM,P_WM)
    mean = (a1+a2+a3)/3
    accurcy = [[a1, a2,a3, mean], ]
    BK = accuracy(T_BK,P_BK)
    return accurcy,BK

def mutil_asd(y_true, y_pred):
    T_BK,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = asd(T_CSF,P_CSF)
    a2 = asd(T_GM,P_GM)
    a3 = asd(T_WM,P_WM)
    mean = (a1+a2+a3)/3
    asd1 = [[a1,a2,a3, mean], ]
    bk = asd(T_BK,P_BK)
    return asd1,bk
def mutil_assd(y_true, y_pred):
    T_BK,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = assd(T_CSF,P_CSF)
    a2 = assd(T_GM,P_GM)
    a3 = assd(T_WM,P_WM)
    mean = (a1+a2+a3)/3
    assd1 = [[a1,a2,a3, mean], ]
    bk = assd(T_BK,P_BK)
    return assd1,bk

def mutil_hd(y_true, y_pred):
    T_BK,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = hd(T_CSF,P_CSF)
    a2 = hd(T_GM,P_GM)
    a3 = hd(T_WM,P_WM)
    mean = (a1+a2+a3)/3
    hd1 = [[a1, a2,a3, mean], ]
    bk = hd(T_BK,P_BK)
    return hd1,bk

def mutil_hd95(y_true, y_pred):
    T_BK,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a1 = hd95(T_CSF,P_CSF)
    a2 = hd95(T_GM,P_GM)
    a3 = hd95(T_WM,P_WM)
    mean = (a1+a2+a3)/3
    hd1 = [[a1, a2,a3, mean], ]
    bk = hd95(T_BK,P_BK)
    return hd1,bk


