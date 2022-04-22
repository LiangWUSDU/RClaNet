"""
This is a tutorial example.
"""
import tensorflow as tf
import labelreg.utils as util


def warp_volumes_by_ddf(input_, ddf):
    grid_warped = util.get_reference_grid(ddf.shape[1:4]) + ddf
    warped = util.resample_linear(tf.convert_to_tensor(input_, dtype=tf.float32), grid_warped)
    with tf.Session() as sess:
        return sess.run(warped)
def compute_binary_dice_val(input1, input2):
    dice_val=util.compute_binary_dice(tf.convert_to_tensor(input1, dtype=tf.float32), tf.convert_to_tensor(input2, dtype=tf.float32))
    with tf.Session() as sess:
        return sess.run(dice_val)
def compute_centroid_distance_val(input1, input2, grid=None):
    centroid_distance=util.compute_centroid_distance(tf.convert_to_tensor(input1, dtype=tf.float32), tf.convert_to_tensor(input2, dtype=tf.float32), grid=None)
    with tf.Session() as sess:
        return sess.run(centroid_distance)