import tensorflow as tf
import keras.backend as K
import numpy as np
from functions.loss import *

def build_loss(label_moving, label_fixed, ddf, similarity_type, similarity_scales, regulariser_type, regulariser_weight,
			   network_type='local', bidir=None, ddf_inv=None, label_moving_inv=None, label_fixed_inv=None):
	if bidir is None:
		label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales)
	else:
		label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales) + \
						   multi_scale_loss(label_fixed_inv, label_moving_inv, similarity_type.lower(),
											similarity_scales)

	if network_type.lower() == 'global':
		ddf_regularisation = tf.constant(0.0)
	else:
		if bidir is None:
			ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight))
		else:
			ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight)) + \
								 tf.reduce_mean(
									 local_displacement_energy(ddf_inv, regulariser_type, regulariser_weight))
	return tf.reduce_mean(label_similarity), ddf_regularisation


def build_loss_t1t2(similarity_type, similarity_scales, regulariser_type, regulariser_weight,
					label_moving, label_moving_t1, label_moving_t2, label_fixed, network_type, ddf, ddf_t1, ddf_t2):
	label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales)
	label_similarity_t1 = multi_scale_loss(label_fixed, label_moving_t1, similarity_type.lower(), similarity_scales)
	label_similarity_t2 = multi_scale_loss(label_fixed, label_moving_t2, similarity_type.lower(), similarity_scales)
	label_similarity_total = label_similarity + label_similarity_t1 + label_similarity_t2
	if network_type.lower() == 'global':
		ddf_regularisation = tf.constant(0.0)
	else:
		ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight)) + \
							 tf.reduce_mean(local_displacement_energy(ddf_t1, regulariser_type, regulariser_weight)) + \
							 tf.reduce_mean(local_displacement_energy(ddf_t2, regulariser_type, regulariser_weight)) + \
							 tf.reduce_mean(
								 tf.square(ddf - ddf_t1) + tf.square(ddf - ddf_t2) + tf.square(ddf_t1 - ddf_t2))
	return tf.reduce_mean(label_similarity_total), ddf_regularisation

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
def build_loss_mask(similarity_type, similarity_scales, regulariser_type, regulariser_weight,
					label_moving, label_fixed, network_type, ddf, mask=None):
	label_similarity = multi_scale_loss(label_fixed, label_moving, similarity_type.lower(), similarity_scales)
	if network_type.lower() == 'global':
		ddf_regularisation = tf.constant(0.0)
	else:
		ddf_regularisation = tf.reduce_mean(local_displacement_energy(ddf, regulariser_type, regulariser_weight)) + \
							 tf.reduce_mean(local_displacement_energy(ddf * mask, regulariser_type, regulariser_weight))
	return tf.reduce_mean(label_similarity), ddf_regularisation


def weighted_binary_cross_entropy(ts, ps, pw=1, eps=1e-6):
	ps = tf.clip_by_value(ps, eps, 1 - eps)
	return -tf.reduce_sum(
		tf.concat([ts * pw, 1 - ts], axis=4) * tf.log(tf.concat([ps, 1 - ps], axis=4)),
		axis=4, keep_dims=True)


def dice_simple(ts, ps, eps_vol=1e-6):
	numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3, 4]) * 2
	denominator = tf.reduce_sum(ts, axis=[1, 2, 3, 4]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4]) + eps_vol
	return numerator / denominator


def double_dice_simple(ts, ps, eps_vol=1e-6):
	numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3, 4]) * 2 - tf.reduce_sum((1 - ts) * ps,
																			  axis=[1, 2, 3, 4]) - tf.reduce_sum(
		ts * (1 - ps), axis=[1, 2, 3, 4])
	denominator = tf.reduce_sum(ts, axis=[1, 2, 3, 4]) + tf.reduce_sum(ps, axis=[1, 2, 3, 4]) + eps_vol
	return numerator / denominator  # added for outside the overlap

def generalized_dice_loss(y_true, y_pred):
    '''
    https://arxiv.org/pdf/1707.03237.pdf
    '''
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2,3))  # Count the number of pixels in the target area
    w = 1/(w**2+0.000001) # Calculate category weights
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3,4))
    numerator = K.sum(numerator)  #molecular


    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3,4))
    denominator = K.sum(denominator)  #denominator

    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def categorical_loss(y_true, y_pred):
    loss =  K.categorical_crossentropy(y_pred , y_true)
    return loss

def ssim_simple(ts, ps):
	ssim_v = tf.reduce_sum(tf.image.ssim(ts, ps, 1.0))
	return ssim_v

def ssim_simple_mean(ts, ps):
    ssim_v = tf.reduce_mean(tf.image.ssim(ts, ps, 1.0))
    return ssim_v
def ssim_multiscal(ts, ps):
	ssim_v = tf.reduce_sum(tf.image.ssim_multiscale(ts, ps, 1.0))
	return ssim_v


def dice_generalised(ts, ps, weights):
	ts2 = tf.concat([ts, 1 - ts], axis=4)
	ps2 = tf.concat([ps, 1 - ps], axis=4)
	numerator = 2 * tf.reduce_sum(tf.reduce_sum(ts2 * ps2, axis=[1, 2, 3]) * weights, axis=1)
	denominator = tf.reduce_sum((tf.reduce_sum(ts2, axis=[1, 2, 3]) +
								 tf.reduce_sum(ps2, axis=[1, 2, 3])) * weights, axis=1)
	return numerator / denominator


def jaccard_simple(ts, ps, eps_vol=1e-6):
	numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3, 4])
	denominator = tf.reduce_sum(tf.square(ts), axis=[1, 2, 3, 4]) + \
				  tf.reduce_sum(tf.square(ps), axis=[1, 2, 3, 4]) - numerator + eps_vol
	return numerator / denominator


def gauss_kernel1d(sigma):
	if sigma == 0:
		return 0
	else:
		tail = int(sigma * 3)
		k = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
		return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma):  # this is an approximation
	if sigma == 0:
		return 0
	else:
		tail = int(sigma * 5)
		# k = tf.reciprocal(([((x/sigma)**2+1)*sigma*3.141592653589793 for x in range(-tail, tail+1)]))
		k = tf.reciprocal([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
		return k / tf.reduce_sum(k)


def separable_filter3d(vol, kernel):
	if kernel == 0:
		return vol
	else:
		strides = [1, 1, 1, 1, 1]
		if vol.get_shape()[-1] == 1:
			return tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
				vol,
				tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
				tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
				tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME")
		else:
			return tf.concat([tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
				tf.expand_dims(vol[..., i], -1),
				tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
				tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
				tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME") for i in range(0, 3)], -1)


def single_scale_loss(label_fixed, label_moving, loss_type):
	if loss_type == 'cross-entropy':
		label_loss_batch = tf.reduce_mean(weighted_binary_cross_entropy(label_fixed, label_moving), axis=[1, 2, 3, 4])
	elif loss_type == 'mean-squared':
		label_loss_batch = tf.reduce_mean(tf.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4])
	elif loss_type == 'dice':
		label_loss_batch = 1 - dice_simple(label_fixed, label_moving)
	elif loss_type == 'doubledice':
		label_loss_batch = 1 - double_dice_simple(label_fixed, label_moving)
	elif loss_type == 'doubledice-ssim':
		label_loss_batch = 1 - double_dice_simple(label_fixed, label_moving) - ssim_simple(label_fixed, label_moving)
	elif loss_type == 'doubledice-ssim-mean':
		label_loss_batch = 1 - double_dice_simple(label_fixed, label_moving) - ssim_simple_mean(label_fixed,label_moving)
	elif loss_type == 'ssim-mean-mse':
		label_loss_batch =  1- ssim_simple_mean(label_fixed,label_moving) + tf.reduce_mean(tf.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4])
	elif loss_type == 'jaccard':
		label_loss_batch = 1 - jaccard_simple(label_fixed, label_moving)
	elif loss_type == 'ssim-distance':
		label_loss_batch = 1 - ssim_simple_mean(label_fixed,label_moving)+0.01*tf.sqrt(tf.reduce_sum(tf.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4]))
	else:
		raise Exception('Not recognised label correspondence loss! loss_type:' + loss_type)
	return label_loss_batch

def multi_scale_loss(label_fixed, label_moving, loss_type,loss_scales):
	label_loss_all = tf.stack(
		[single_scale_loss(
			separable_filter3d(label_fixed, gauss_kernel1d(s)),
			separable_filter3d(label_moving, gauss_kernel1d(s)), loss_type)
			for s in loss_scales],
		axis=1)
	return tf.reduce_mean(label_loss_all, axis=1)


def local_displacement_energy(ddf, energy_type, energy_weight):
	#    def gradient_dx(fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2
	#
	#    def gradient_dy(fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2
	#
	#    def gradient_dz(fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2
	#
	#    def gradient_txyz(Txyz, fn):
	#        return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=4)
	#        return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=-1)
	def gradient_dx(fv):
		return (fv[:, 2:, 1:-1, 1:-1, :] - fv[:, :-2, 1:-1, 1:-1, :]) / 2

	def gradient_dy(fv):
		return (fv[:, 1:-1, 2:, 1:-1, :] - fv[:, 1:-1, :-2, 1:-1, :]) / 2

	def gradient_dz(fv):
		return (fv[:, 1:-1, 1:-1, 2:, :] - fv[:, 1:-1, 1:-1, :-2, :]) / 2

	def gradient_txyz(Txyz, fn):
		return fn(Txyz)

	def compute_gradient_norm(displacement, flag_l1=False):
		dTdx = gradient_txyz(displacement, gradient_dx)
		dTdy = gradient_txyz(displacement, gradient_dy)
		dTdz = gradient_txyz(displacement, gradient_dz)
		if flag_l1:
			norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
		else:
			norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
		return tf.reduce_mean(norms, [1, 2, 3, 4])

	def compute_bending_energy(displacement):
		dTdx = gradient_txyz(displacement, gradient_dx)
		dTdy = gradient_txyz(displacement, gradient_dy)
		dTdz = gradient_txyz(displacement, gradient_dz)
		#        [ndim0,ndim1,ndim2,ndim3,ndim4]= dTdx.get_shape().as_list()
		#        ndims = len(dTdx.get_shape().as_list()) - 2
		#        assert ndims in [3], "volumes should be 1 to 3 dimensions. found: %d,%s,%s,%s,%s,%s" %(ndims, ndim0,ndim1,ndim2,ndim3,ndim4)
		#        [ndim0,ndim1,ndim2,ndim3,ndim4]= dTdy.get_shape().as_list()
		#        ndims = len(dTdy.get_shape().as_list()) - 2
		#        assert ndims in [3], "volumes should be 1 to 3 dimensions. found: %d,%s,%s,%s,%s,%s" %(ndims, ndim0,ndim1,ndim2,ndim3,ndim4)
		#        [ndim0,ndim1,ndim2,ndim3,ndim4]= dTdz.get_shape().as_list()
		#        ndims = len(dTdz.get_shape().as_list()) - 2
		#        assert ndims in [3], "volumes should be 1 to 3 dimensions. found: %d,%s,%s,%s,%s,%s" %(ndims, ndim0,ndim1,ndim2,ndim3,ndim4)
		dTdxx = gradient_txyz(dTdx, gradient_dx)
		dTdyy = gradient_txyz(dTdy, gradient_dy)
		dTdzz = gradient_txyz(dTdz, gradient_dz)
		dTdxy = gradient_txyz(dTdx, gradient_dy)
		dTdyz = gradient_txyz(dTdy, gradient_dz)
		dTdxz = gradient_txyz(dTdx, gradient_dz)
		#        dTdxz = gradient_txyz(dTdz, gradient_dx)
		return tf.reduce_mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2,
							  [1, 2, 3, 4])

	if energy_weight:
		if energy_type == 'bending':
			energy = compute_bending_energy(ddf)
		elif energy_type == 'gradient-l2':
			energy = compute_gradient_norm(ddf)
		elif energy_type == 'gradient-l1':
			energy = compute_gradient_norm(ddf, flag_l1=True)
		else:
			raise Exception('Not recognised local regulariser! energy_type:' + energy_type)
	else:
		energy = tf.constant(0.0)

	return energy * energy_weight


class Miccai2018():
	"""
	N-D main loss for VoxelMorph MICCAI Paper
	prior matching (KL) term + image matching term
	"""

	# def __init__(self, flow_mean, flow_log_sigma, image_sigma=None, prior_lambda=2, flow_vol_shape=None):
	# self.flow_mean=flow_mean
	# self.flow_log_sigma=flow_log_sigma
	def __init__(self, image_sigma=None, prior_lambda=2, flow_vol_shape=None):
		self.image_sigma = image_sigma
		self.prior_lambda = prior_lambda
		self.D = None
		self.flow_vol_shape = flow_vol_shape

	def _adj_filt(self, ndims):
		"""
		compute an adjacency filter that, for each feature independently,
		has a '1' in the immediate neighbor, and 0 elsewehre.
		so for each filter, the filter has 2^ndims 1s.
		the filter is then setup such that feature i outputs only to feature i
		"""

		# inner filter, that is 3x3x...
		filt_inner = np.zeros([3] * ndims)
		for j in range(ndims):
			o = [[1]] * ndims
			o[j] = [0, 2]
			filt_inner[np.ix_(*o)] = 1

		# full filter, that makes sure the inner filter is applied
		# ith feature to ith feature
		filt = np.zeros([3] * ndims + [ndims, ndims])
		for i in range(ndims):
			filt[..., i, i] = filt_inner

		return filt

	def _degree_matrix(self, vol_shape):
		# get shape stats
		ndims = len(vol_shape)
		sz = [*vol_shape, ndims]

		# prepare conv kernel
		conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

		# prepare tf filter
		z = K.ones([1] + sz)
		filt_tf = tf.convert_to_tensor(self._adj_filt(ndims), dtype=tf.float32)
		strides = [1] * (ndims + 2)
		return conv_fn(z, filt_tf, strides, "SAME")

	def prec_loss(self, y_pred):
		"""
		a more manual implementation of the precision matrix term
				mu * P * mu    where    P = D - A
		where D is the degree matrix and A is the adjacency matrix
				mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
		where j are neighbors of i

		Note: could probably do with a difference filter,
		but the edges would be complicated unless tensorflow allowed for edge copying
		"""
		vol_shape = y_pred.get_shape().as_list()[1:-1]
		ndims = len(vol_shape)

		sm = 0
		for i in range(ndims):
			d = i + 1
			# permute dimensions to put the ith dimension first
			r = [d, *range(d), *range(d + 1, ndims + 2)]
			y = K.permute_dimensions(y_pred, r)
			df = y[1:, ...] - y[:-1, ...]
			sm += K.mean(df * df)

		return 0.5 * sm / ndims

	def kl_loss(self, mean, log_sigma):
		"""
		KL loss
		y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
		D (number of dimensions) should be 1, 2 or 3

		y_true is only used to get the shape
		"""

		# prepare inputs
		# ndims = len(y_pred.get_shape()) - 2
		# mean = self.flow_mean
		# log_sigma = self.flow_log_sigma
		ndims = len(mean.get_shape()) - 2
		if self.flow_vol_shape is None:
			# Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
			self.flow_vol_shape = mean.get_shape().as_list()[1:-1]

		# compute the degree matrix (only needs to be done once)
		# we usually can't compute this until we know the ndims,
		# which is a function of the data
		if self.D is None:
			self.D = self._degree_matrix(self.flow_vol_shape)

		# sigma terms
		sigma_term = self.prior_lambda * self.D * tf.exp(log_sigma) - log_sigma
		sigma_term = K.mean(sigma_term)

		# precision terms
		# note needs 0.5 twice, one here (inside self.prec_loss), one below
		prec_term = self.prior_lambda * self.prec_loss(mean)

		# combine terms
		return 0.5 * ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

	def recon_loss(self, y_true, y_pred):
		""" reconstruction loss """
		return 1. / (self.image_sigma ** 2) * K.mean(K.square(y_true - y_pred))

