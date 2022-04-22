
# Third party inports
import tensorflow as tf
import numpy as np
import scipy.io as sio
# third party
from scipy.interpolate import interpn
# project
import sys
import keras.backend as K

from functions.ext.medipylib.medipy.metrics import dice
# batch_sizexheightxwidthxdepthxchan


def diceLoss(y_true, y_pred):
    top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
    bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice

def CalcDice(vol_size, labels,Y_flow,atlas_seg,X_seg):
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    # Warp segments with flow
    flow = Y_flow[0][ :, :, :, :]
    sample = flow+grid
    sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    warp_segY1 = interpn((yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)
    valsY_T1, valsY_T1_labels = dice(warp_segY1, atlas_seg[0, :, :, :, 0], labels=labels, nargout=2)
    #print("valsY_T1 dice:"+str(valsY_T1)+str(valsY_T1_labels))
    returnval = [valsY_T1]
    returnval.append(valsY_T1_labels)
    return tuple(returnval)


def gradientLoss1(penalty='l1'):
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        d = (tf.reduce_sum(dx)+tf.reduce_sum(dy)+tf.reduce_sum(dz))/tf.to_float(tf.count_nonzero(y_pred))
        return d/3.0
    return loss

def gradient_cc3D_Loss(win=[9, 9, 9], penalty='l1',voxel_weights=None):
    def lossCC(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights
        return -1.0*tf.reduce_sum(cc)/tf.to_float(tf.count_nonzero(cc))
        #return -1.0*tf.reduce_mean(cc)
    def Mse(I,J):
        return K.mean(K.square(I - J), axis=-1)

    def lossGradientCC(I, J):
        ccImage = lossCC(I,J)
        Idy = tf.abs(I[:, 1:, :, :, :] - I[:, :-1, :, :, :])
        Idx = tf.abs(I[:, :, 1:, :, :] - I[:, :, :-1, :, :])
        Idz = tf.abs(I[:, :, :, 1:, :] - I[:, :, :, :-1, :])
        Jdy = tf.abs(J[:, 1:, :, :, :] - J[:, :-1, :, :, :])
        Jdx = tf.abs(J[:, :, 1:, :, :] - J[:, :, :-1, :, :])
        Jdz = tf.abs(J[:, :, :, 1:, :] - J[:, :, :, :-1, :])
        dx  = tf.abs(Idx - Jdx)
        dy  = tf.abs(Idy - Jdy)
        dz  = tf.abs(Idz - Jdz)
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        #        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        d = (tf.reduce_sum(dx)+tf.reduce_sum(dy)+tf.reduce_sum(dz))/tf.to_float(tf.count_nonzero(I))
        return d/3.0+ ccImage+Mse(I,J)
    return lossGradientCC

def cc3D(win=[9, 9, 9], voxel_weights=None):
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights
        return -1.0*tf.reduce_sum(cc)/tf.to_float(tf.count_nonzero(cc))
        #return -1.0*tf.reduce_mean(cc)

    return loss
def Mse(I,J):
    return K.mean(K.square(I - J), axis=-1)

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def ncc( I, J,win=[9, 9, 9],eps=1e-5):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(I.get_shape().as_list()) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
    if win is None:
        win = [9] * ndims

        # get convolution function
    conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
    I2 = I*I
    J2 = J*J
    IJ = I*J

        # compute filters
    sum_filt = tf.ones([win[0], win[1], win[2], 1, 1])
    strides = [1] * (ndims + 2)
    padding = 'SAME'

        # compute local sums via convolution
    I_sum = conv_fn(I, sum_filt, strides, padding)
    J_sum = conv_fn(J, sum_filt, strides, padding)
    I2_sum = conv_fn(I2, sum_filt, strides, padding)
    J2_sum = conv_fn(J2, sum_filt, strides, padding)
    IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
    win_size = np.prod(win)
    u_I = I_sum/win_size
    u_J = J_sum/win_size

    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

    cc = cross*cross / (I_var*J_var + eps)

        # return negative cc.
    return tf.reduce_mean(cc)

def loss(I, J):
    return - ncc(I, J)+Mse(I,J)
def ncc_loss(I, J):
    return - ncc(I, J)
def ncchuberloss(I,J):
    return -ncc(I,J)+ tf.losses.huber_loss(I,J,delta=0.1)
def ssim_simple_mean(ts, ps):
    ssim_v = tf.reduce_mean(tf.image.ssim(ts, ps, 1.0))
    return ssim_v
def ncchuber_ssim_loss(I,J):
    return -ncc(I,J)+ tf.losses.huber_loss(I,J,delta=0.1)-ssim_simple_mean(I,J)
def gradientLoss(penalty='l1'):
    # scale = tf.constant([[par['res2']-1, 0, 0], [0, par['res1']-1, 0], [0,0,par['res3']-1]], dtype=tf.float32)
    def loss(y_true, y_pred):
        # y_pred_O = tf.einsum('abcde,ef->abcdf',y_pred, scale)*0.5

        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dz)
        return d / 3.0

    return loss


def LLoss(penalty='l1'):
    def loss(y_true, y_pred):
        dy = y_pred[:, 0, :, :, :]
        dx = y_pred[:, :, 0, :, :]
        dz = y_pred[:, :, :, 0, :]
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dz)
        return d / 3.0

    return loss


def TvsLoss(penalty='l1', w1=1, w2=0):
    #    scale = 0.5*tf.constant([[par['res2']-1.0, 0, 0], [0, par['res1']-1.0, 0], [0,0,par['res3']-1.0]], dtype=tf.float32)
    def loss(y_true, y_pred):
        #        y_pred_O = tf.einsum('abcde,ef->abcdf',y_pred, scale)

        xx = y_pred[:, :, :, :, 0]  # Possible Problems here
        yy = y_pred[:, :, :, :, 1]
        zz = y_pred[:, :, :, :, 2]

        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

            yy = yy * yy
            xx = xx * xx
            zz = zz * zz

        d = tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dz)
        D = tf.reduce_mean(xx + yy + zz)

        return w1 * d / 3.0 + w2 * D

    return loss


def Get_Ja(displacement):
    '''
    Calculate the Jacobian value at each point of the displacement map having

    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3

    '''

    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])

    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])

    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])

    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])

    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    return D1 - D2 + D3


def NJ_loss(y_true, ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (tf.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return tf.reduce_sum(Neg_Jac)

def gradient_dx(fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

def gradient_dy(fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

def gradient_dz(fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

def gradient_txyz(Txyz, fn):
    return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=4)

def compute_gradient_norm(displacement, flag_l1=False):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    if flag_l1:
        norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
    else:
        norms = dTdx**2 + dTdy**2 + dTdz**2
    return tf.reduce_mean(norms, [1, 2, 3, 4])

def compute_bending_energy(y_true,displacement):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    dTdxx = gradient_txyz(dTdx, gradient_dx)
    dTdyy = gradient_txyz(dTdy, gradient_dy)
    dTdzz = gradient_txyz(dTdz, gradient_dz)
    dTdxy = gradient_txyz(dTdx, gradient_dy)
    dTdyz = gradient_txyz(dTdy, gradient_dz)
    dTdxz = gradient_txyz(dTdx, gradient_dz)
    return tf.reduce_mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2, [1, 2, 3, 4])

def reg_loss(y_true, y_pred,w = 1e-3):
    return gradientLoss('l2')(y_true, y_pred) + w*NJ_loss(y_true, y_pred)

def reg_loss1(y_true, y_pred,w1= 0.5, w = 1e-3):
    return w1*gradientLoss('l2')(y_true, y_pred) + w*NJ_loss(y_true, y_pred)

def local_displacement_energy(ddf, energy_type, energy_weight):

    def gradient_dx(fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(Txyz, fn):
        return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=4)

    def compute_gradient_norm(displacement, flag_l1=False):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        if flag_l1:
            norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return tf.reduce_mean(norms, [1, 2, 3, 4])

    def compute_bending_energy(displacement):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        dTdxx = gradient_txyz(dTdx, gradient_dx)
        dTdyy = gradient_txyz(dTdy, gradient_dy)
        dTdzz = gradient_txyz(dTdz, gradient_dz)
        dTdxy = gradient_txyz(dTdx, gradient_dy)
        dTdyz = gradient_txyz(dTdy, gradient_dz)
        dTdxz = gradient_txyz(dTdx, gradient_dz)
        return tf.reduce_mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2, [1, 2, 3, 4])

    if energy_weight:
        if energy_type == 'bending':
            energy = compute_bending_energy(ddf)
        elif energy_type == 'gradient-l2':
            energy = compute_gradient_norm(ddf)
        elif energy_type == 'gradient-l1':
            energy = compute_gradient_norm(ddf, flag_l1=True)
        elif energy_type == 'bending-nj':
            energy = compute_bending_energy(ddf)+NJ_loss(ddf)
        elif energy_type == 'bending-njmean':
            energy = compute_bending_energy(ddf)+NJmean_loss(ddf)
        elif energy_type == 'bending-nj2':
            energy = compute_bending_energy(ddf)+NJ2_loss(ddf)
        elif energy_type == 'bending-nj-j1':
            energy = compute_bending_energy(ddf)+NJ_loss(ddf)+J1_loss(ddf)
        elif energy_type == 'bending-nj2-j1mean':
            energy = compute_bending_energy(ddf)+NJ_loss(ddf)+J1mean_loss(ddf)
        elif energy_type == 'bending-njmean-j1mean':
            energy =  compute_bending_energy(ddf)+NJmean_loss(ddf)+J1mean_loss(ddf)
        else:
            raise Exception('Not recognised local regulariser!')
    else:
        energy = tf.constant(0.0)

    return energy*energy_weight