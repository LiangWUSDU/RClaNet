from keras.models import Model
from keras.layers import Conv3D,Input, concatenate,Conv3DTranspose,add,BatchNormalization
from keras.layers import MaxPooling3D,Flatten,Dense,Lambda,Dropout, LeakyReLU,Reshape,Add
from keras.initializers import RandomNormal
from functions.ext.neuron.layers import SpatialTransformer
import keras.layers as KL
from keras import backend as K
from functions.voxmorphloss import multi_scale_loss
from networks.model_functions import myConv,encoder,decoder_SAM_multiscale,BatchActivate
import numpy as np
from keras.engine.topology import Layer

#def Separ(x,filters ):
#    x1 = SeparableConv2D(filters,(3,3),padding='same')(x)
#    x1 = LeakyReLU()(x1)
#    x1 = SeparableConv2D(filters,(3,3),padding='same')(x1)
#    x1 = LeakyReLU()(x1)
#    y =  SeparableConv2D(filters,(1,1),padding='same')(x)
#    y = BatchNormalization()(y)
#    y = LeakyReLU()(y)
#    z = add([x1,y])
#    return z
def Conv3d_BN(x, nb_filter, kernel_size, dilation_rate, padding='same'):
	x = Conv3D(nb_filter, kernel_size, padding=padding, dilation_rate=dilation_rate)(x)
	y = BatchNormalization()(x)
	y = LeakyReLU()(y)
	return y
def identity_Block(inpt, nb_filter, kernel_size, dilation_rate, with_conv_shortcut=False):
	x = Conv3d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')
	x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,dilation_rate=dilation_rate, padding='same')
	if with_conv_shortcut:
		shortcut = Conv3d_BN(inpt, nb_filter=nb_filter, dilation_rate=dilation_rate,kernel_size=kernel_size)
		x = Dropout(0.2)(x)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, inpt])
		return x
def Dresnet(vol_size,fl_nf,src_feats=1):
	inputs_T1 = Input(shape=[*vol_size, src_feats])
	x1 = MaxPooling3D((2,2,2))(inputs_T1)
	x1 = identity_Block(x1, nb_filter=fl_nf[0], kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
	x2 = identity_Block(x1, nb_filter=fl_nf[0], kernel_size=(3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
	x = concatenate([x1,x2],axis=-1)
	x = MaxPooling3D((2,2,2))(x)  ##80
	y1 = identity_Block(x, nb_filter=fl_nf[1], kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
	y2 = identity_Block(y1, nb_filter=fl_nf[1], kernel_size=(3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
	y = concatenate([y1,y2],axis=-1)
	y = MaxPooling3D((2,2,2))(y) ##40
	z1 = identity_Block(y, nb_filter=fl_nf[2], kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
	z2 = identity_Block(z1, nb_filter=fl_nf[2], kernel_size=(3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
	z = concatenate([z1,z2],axis=-1)
	z = MaxPooling3D((2,2,2))(z) ## 20
	k1 = identity_Block(z, nb_filter=fl_nf[3], kernel_size=(3, 3,3), dilation_rate=(1,1,1), with_conv_shortcut=True)
	k2 = identity_Block(k1, nb_filter=fl_nf[3], kernel_size=(3, 3,3), dilation_rate=(3,3,3), with_conv_shortcut=True)
	k = concatenate([k1,k2],axis=-1)
	k = MaxPooling3D((4,4,4))(k) ## 20
	k = Flatten()(k)
	k = Dense(fl_nf[4],activation='relu')(k)
	k = Dense(fl_nf[5],activation='relu')(k)
	outputs = Dense(fl_nf[6],activation='softmax')(k)
	model = Model(inputs=inputs_T1, outputs=outputs)
	return model
def CNN_7(vol_size,fl_nf,src_feats=1):
	inputs_T1 = Input(shape=[*vol_size, src_feats])
	## 1
	x = myConv(inputs_T1,fl_nf[0] , strides=1)
	x = Dropout(0.5)(x)
	x = myConv(x,fl_nf[0] , strides=2) ##80
	## 2
	x = myConv(x,fl_nf[1] , strides=1)
	x =  myConv(x,fl_nf[1] , strides=2) ##40
	## 3
	x = myConv(x,fl_nf[2] , strides=1)
	x = myConv(x,fl_nf[2] , strides=2) ## 20
	## 4
	x = myConv(x,fl_nf[3] , strides=1)
	x = myConv(x,fl_nf[3] , strides=2)  ## 10
	## 5
	x = myConv(x,fl_nf[4] , strides=1)
	x = myConv(x,fl_nf[4] , strides=2) ##  5
	## 6
	x = myConv(x,fl_nf[5] , strides=1)
	x = Dropout(0.5)(x)
	x = myConv(x,fl_nf[5] , strides=2)  ## 2
	##
	x = Flatten()(x)
	x = Dense(fl_nf[6],activation='relu')(x)
	x = Dense(fl_nf[7],activation='relu')(x)
	outputs = Dense(fl_nf[8],activation='sigmoid')(x)
	model = Model(inputs=inputs_T1, outputs=outputs)
	return model

#model = CNN_7((160,192,160),fl_nf=[16,32,64,128,256,512,1024,512,3])
#model.summary()
def MultiHeadsAttModel(l=8 * 8, d=512, dv=64, dout=512, nv=8):
	v1 = Input(shape=(l, d))  ##value  64,512
	q1 = Input(shape=(l, d))  ##query
	k1 = Input(shape=(l, d))  ##key

	v2 = Dense(dv * nv, activation="relu")(v1)  ##64*8 = 512
	q2 = Dense(dv * nv, activation="relu")(q1)
	k2 = Dense(dv * nv, activation="relu")(k1)

	v = Reshape([l, nv, dv])(v2)
	q = Reshape([l, nv, dv])(q2)
	k = Reshape([l, nv, dv])(k2)

	att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv),
				 output_shape=(l, nv, nv))([q, k])  # l, nv, nv
	att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)

	out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 3]), output_shape=(l, nv, dv))([att, v])
	out = Reshape([l, d])(out)

	out = Add()([out, q1])

	out = Dense(dout, activation="relu")(out)

	return Model(inputs=[q1, k1, v1], outputs=out)
class NormL(Layer):

	def __init__(self, **kwargs):
		super(NormL, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.a = self.add_weight(name='kernel',
								 shape=(1,input_shape[-1]),
								 initializer='ones',
								 trainable=True)
		self.b = self.add_weight(name='kernel',
								 shape=(1,input_shape[-1]),
								 initializer='zeros',
								 trainable=True)
		super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		eps = 0.000001
		mu = K.mean(x, keepdims=True, axis=-1)
		sigma = K.std(x, keepdims=True, axis=-1)
		ln_out = (x - mu) / (sigma + eps)
		return ln_out*self.a + self.b

	def compute_output_shape(self, input_shape):
		return input_shape

def mutilheadattModel(vol_size,src_feats=1):
	inputs_T1 = Input(shape=[*vol_size, src_feats])
	x = Conv3D(16, (3, 3, 3), activation='tanh', padding='same')(inputs_T1)
	x = MaxPooling3D(pool_size=(2, 2, 2))(x)
	x = Conv3D(16, (3, 3, 3), activation='tanh', padding='same')(x)
	x = MaxPooling3D(pool_size=(2, 2, 2))(x)
	x = Conv3D(32, (3, 3, 3), activation='tanh', padding='same')(x)   ##40,48,40,192
	x = MaxPooling3D(pool_size=(2, 2, 2))(x)
	x = Conv3D(64 * 3, (3, 3, 3), activation='tanh', padding='same')(x)   ##40,48,40,192
##
	x = Reshape([20*24*20, 64 * 3])(x)  ##6*6 大小  64*3 滤波器个数
	att = MultiHeadsAttModel(l=20*24*20, d=64 * 3, dv=8 * 3, dout=32, nv=8)
	x = att([x, x, x])
	x = Reshape([20,24,20, 32])(x)
	x = BatchNormalization()(x)
##
	x = Flatten()(x)
	x = Dense(256, activation='tanh')(x)
	x = Dropout(0.5)(x)
	outputs = Dense(1, activation='sigmoid')(x)
	model = Model(inputs=inputs_T1, outputs=outputs)
	return model

def MLP(vol_size,fl_nf,src_feats=1):
	inputs_T1 = Input(shape=[*vol_size, src_feats])
	## 1
	x = myConv(inputs_T1,fl_nf[0] , strides=1)
	x = Dropout(0.5)(x)
	x = myConv(x,fl_nf[0] , strides=1) ##80
	## 2
	x = MaxPooling3D((2,2,2))(x)
	x = myConv(x,fl_nf[1] , strides=1)
	x =  myConv(x,fl_nf[1] , strides=1) ##40
	x = MaxPooling3D((2,2,2))(x)
	## 3
	x = myConv(x,fl_nf[2] , strides=1)
	x = myConv(x,fl_nf[2] , strides=1) ## 20
	x = MaxPooling3D((2,2,2))(x)
	## 4
	x = myConv(x,fl_nf[3] , strides=1)
	x = myConv(x,fl_nf[3] , strides=1)  ## 10
	x = MaxPooling3D((2,2,2))(x)
	## 5
	x = myConv(x,fl_nf[4] , strides=1)
	x = myConv(x,fl_nf[4] , strides=1) ##  5
	x = MaxPooling3D((2,2,2))(x)
	## 6
	x = myConv(x,fl_nf[5] , strides=1)
	x = Dropout(0.5)(x)
	x = myConv(x,fl_nf[5] , strides=2)  ## 2
	##
	x = Flatten()(x)
	x = Dense(fl_nf[6],activation='relu')(x)
	x = Dense(fl_nf[7],activation='relu')(x)
	outputs = Dense(fl_nf[8],activation='sigmoid')(x)
	model = Model(inputs=inputs_T1, outputs=outputs)
	return model