from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import MaxPooling3D,Flatten,Dense,Lambda,Dropout, LeakyReLU,Reshape,Add
a = K.ones((4,150,8,24))
b = K.ones((4,150,8,24))
c = K.batch_dot(a, b,axes=[-2, -2])/np.sqrt(24)
d = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-2, 2]) / np.sqrt(24))([a, b])  # l, nv, nv ##150,24,24
print(d.shape)

#import tensorflow as tf
#a = tf.ones((150,8,24))
#b = tf.ones((150,8,24))
#c = tf.matmul(a, b)
#print(c.shape) #(9, 8, 7, 4, 5)