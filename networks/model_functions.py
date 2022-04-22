"""
This are some functions of build model
"""
# third party
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate,Conv3DTranspose,Dropout
from keras.layers import LeakyReLU, Reshape,MaxPooling3D,ReLU,Multiply, Lambda,GlobalAveragePooling3D,GlobalMaxPooling3D,AveragePooling3D,Flatten,Dense,BatchNormalization,add,GlobalAveragePooling3D,multiply
from keras.initializers import RandomNormal
from functions.ext.neuron.layers import SpatialTransformer
import keras
import keras.layers as KL
import numpy as np
from functions.voxmorphloss import multi_scale_loss
from keras import backend as K

def BatchActivate(x,name = None):
    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
    x = LeakyReLU(0.2)(x)
    return x
def myConv(x_in, nf, strides=1):
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=1)(x_in)
    x_out = BatchActivate(x_out)
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_out)
    x_out = BatchActivate(x_out)
    shortcut = Conv3D(nf, kernel_size=(1, 1, 1), strides=strides, kernel_initializer='he_normal')(x_in)
    x_out = add([shortcut, x_out])
    return x_out

def se_block(x, filters, ratio=16):
    """
    creates a squeeze and excitation block
    https://arxiv.org/abs/1709.01507

    Parameters
    ----------
    x : tensor
        Input keras tensor
    ratio : int
        The reduction ratio. The default is 16.
    Returns
    -------
    x : tensor
        A keras tensor
    """

    se_shape = (1,1,1, filters)

    se = GlobalAveragePooling3D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([x, se])
    return x
def SAM(F1,F2,filter):
    ##F1为encoder路径的特征图
    ##F2为decoder路径的特征图
    ##filter为encoder路径的大小
    Max = MaxPooling3D()(F2)
    Avg = AveragePooling3D()(F2)
    MA = concatenate([Max,Avg],axis=-1)
    d3 = Conv3D(filter,kernel_size=(3,3,3),dilation_rate=(1,1,1),activation='relu',padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(MA)
    d5 = Conv3D(filter,kernel_size=(3,3,3),dilation_rate=(2,2,2),activation='relu',padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(MA)
    d7 = Conv3D(filter,kernel_size=(3,3,3),dilation_rate=(5,5,5),activation='relu',padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(MA)
    d = concatenate([d3,d5,d7],axis=-1)
    du = UpSampling3D((2,2,2))(d)
    A2 = Conv3D(filter,kernel_size=(1,1,1),activation='sigmoid',padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(du)
    A = multiply([A2,F1])
    return A

def encoder(x,enc_nf):
    to_decoder = []
    main_path = myConv(x, enc_nf[0], 1)
    to_decoder.append(main_path)

    main_path = myConv(main_path, enc_nf[1], 2)
    to_decoder.append(main_path)

    main_path = myConv(main_path, enc_nf[2], 2)
    to_decoder.append(main_path)

    main_path = myConv(main_path, enc_nf[3], 2)
    to_decoder.append(main_path)

    return to_decoder

def decoder_se_block(Brider_path, from_encoder_moving,from_encoder_fixed,dec_nf):
    main_path1 = UpSampling3D(size=(2, 2, 2))(Brider_path)
    main_path1 = concatenate([main_path1, from_encoder_moving[3], from_encoder_fixed[3]], axis=-1)
    main_path1 = se_block(main_path1, 80)
    main_path1 = myConv(main_path1, dec_nf[0])

    main_path2 = UpSampling3D(size=(2, 2, 2))(main_path1)
    main_path2 = concatenate([main_path2, from_encoder_moving[2], from_encoder_fixed[2]], axis=-1)
    main_path2 = se_block(main_path2, 96)
    main_path2 = myConv(main_path2, dec_nf[1])

    main_path3 = UpSampling3D(size=(2, 2, 2))(main_path2)
    main_path3 = concatenate([main_path3, from_encoder_moving[1], from_encoder_fixed[1]], axis=-1)
    main_path3 = se_block(main_path3, 96)
    main_path3 =  myConv(main_path3, dec_nf[2])

    main_path4 = UpSampling3D(size=(2, 2, 2))(main_path3)
    main_path4 = concatenate([main_path4, from_encoder_moving[0], from_encoder_fixed[0]], axis=-1)
    main_path4 = se_block(main_path4, 64)
    main_path4 = myConv(main_path4, dec_nf[3])# 64,64,64,16
    return main_path4

def decoder_se_block_multiscale(Brider_path, from_encoder_moving,dec_nf):
    main_path1 = UpSampling3D(size=(2, 2, 2))(Brider_path)
    main_path1 = concatenate([main_path1, from_encoder_moving[3]], axis=-1)
    main_path1 = se_block(main_path1, 48)
    main_path1 = myConv(main_path1, dec_nf[0])
    #main_path1 = Dropout(0.5)(main_path1)

    main_path2 = UpSampling3D(size=(2, 2, 2))(main_path1)
    main_path2 = concatenate([main_path2, from_encoder_moving[2]], axis=-1)
    main_path2 = se_block(main_path2, 64)
    main_path2 = myConv(main_path2, dec_nf[1])
    #main_path2 = Dropout(0.5)(main_path2)

    main_path3 = UpSampling3D(size=(2, 2, 2))(main_path2)
    main_path3 = concatenate([main_path3, from_encoder_moving[1]], axis=-1)
    main_path3 = se_block(main_path3, 64)
    main_path3 =  myConv(main_path3, dec_nf[2])

    main_path4 = UpSampling3D(size=(2, 2, 2))(main_path3)
    main_path4 = concatenate([main_path4, from_encoder_moving[0]], axis=-1)
    main_path4 = se_block(main_path4, 48)
    main_path4 = myConv(main_path4, dec_nf[3])# 64,64,64,16
    return main_path1,main_path2,main_path3,main_path4

def decoder_SAM_multiscale(Brider_path, from_encoder_moving,enc_nf,dec_nf):
    main_path1 = UpSampling3D(size=(2, 2, 2))(Brider_path)
    main_path1 = SAM(from_encoder_moving[3],main_path1,enc_nf[3])
    main_path1 = myConv(main_path1, dec_nf[0])

    main_path2 = UpSampling3D(size=(2, 2, 2))(main_path1)
    main_path2 = SAM(from_encoder_moving[2],main_path2,enc_nf[2])
    main_path2 = myConv(main_path2, dec_nf[1])

    main_path3 = UpSampling3D(size=(2, 2, 2))(main_path2)
    main_path3 = SAM(from_encoder_moving[1],main_path3,enc_nf[1])
    main_path3 =  myConv(main_path3, dec_nf[2])

    main_path4 = UpSampling3D(size=(2, 2, 2))(main_path3)
    main_path4 = SAM(from_encoder_moving[0],main_path4,enc_nf[0])
    main_path4 = myConv(main_path4, dec_nf[3])# 64,64,64,16

    return main_path1,main_path2,main_path3,main_path4
def decoder_withoutMAM(Brider_path, from_encoder_moving,enc_nf,dec_nf):
    main_path1 = UpSampling3D(size=(2, 2, 2))(Brider_path)
    main_path1 = concatenate([main_path1, from_encoder_moving[3]], axis=-1)
    main_path1 = myConv(main_path1, dec_nf[0])

    main_path2 = UpSampling3D(size=(2, 2, 2))(main_path1)
    main_path2 = concatenate([main_path2, from_encoder_moving[2]], axis=-1)
    main_path2 = myConv(main_path2, dec_nf[1])

    main_path3 = UpSampling3D(size=(2, 2, 2))(main_path2)
    main_path3 = concatenate([main_path3, from_encoder_moving[1]], axis=-1)
    main_path3 =  myConv(main_path3, dec_nf[2])

    main_path4 = UpSampling3D(size=(2, 2, 2))(main_path3)
    main_path4 = concatenate([main_path4, from_encoder_moving[0]], axis=-1)
    main_path4 = myConv(main_path4, dec_nf[3])# 64,64,64,16

    return main_path1,main_path2,main_path3,main_path4






def decoder_multiscale(Brider_path, from_encoder_moving,enc_nf,dec_nf):
    main_path1 = UpSampling3D(size=(2, 2, 2))(Brider_path)
    main_path1 = myConv(main_path1, dec_nf[0])

    main_path2 = UpSampling3D(size=(2, 2, 2))(main_path1)
    main_path2 = myConv(main_path2, dec_nf[1])

    main_path3 = UpSampling3D(size=(2, 2, 2))(main_path2)
    main_path3 =  myConv(main_path3, dec_nf[2])

    main_path4 = UpSampling3D(size=(2, 2, 2))(main_path3)
    main_path4 = myConv(main_path4, dec_nf[3])# 64,64,64,16
    return main_path1,main_path2,main_path3,main_path4
def nn_trf(vol_size, ndim = 1,indexing='xy',interp_method='nearest'):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size,ndim), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    # note the nearest neighbour interpolation method
    # note xy indexing because Guha's original code switched x and y dimensions
    nn_output = SpatialTransformer(interp_method, indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)

def flow_norm(x): ## 在网络训练中计算变形场的模
    return K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
