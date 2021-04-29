import tensorflow.compat.v1 as tf
from postprocess import periodic_post_process, dirichlet_post_process

def black_box_periodic_output_transform(inputs, grid_size):
    idx = [i for i in range(0, grid_size - 1, 2)]
    up_contributions_output = tf.gather(inputs, [i for i in range(0, grid_size, 2)], axis=1)
    up_contributions_output = tf.gather(up_contributions_output, [i for i in range(1, grid_size, 2)], axis=2)
    up_contributions_output = -tf.reduce_sum(up_contributions_output[:, :, :, :, 0], axis=-1) / \
                              tf.reduce_sum(up_contributions_output[:, :, :, :, 1], axis=-1)

    left_contributions_output = tf.gather(inputs, idx, axis=1)
    left_contributions_output = tf.gather(left_contributions_output,
                                          [i for i in range(0, grid_size, 2)], axis=2)
    left_contributions_output = -tf.reduce_sum(left_contributions_output[:, :, :, 2, :],
                                               axis=-1) / tf.reduce_sum(
        left_contributions_output[:, :, :, 1, :], axis=-1)

    right_contributions_output = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    right_contributions_output = tf.gather(right_contributions_output,
                                           [i for i in range(0, grid_size, 2)], axis=2)
    right_contributions_output = -tf.reduce_sum(right_contributions_output[:, :, :, 0, :],
                                                axis=-1) / tf.reduce_sum(
        right_contributions_output[:, :, :, 1, :], axis=-1)
    down_contributions_output = tf.gather(inputs, [i for i in range(0, grid_size, 2)], axis=1)
    down_contributions_output = tf.gather(down_contributions_output, idx, axis=2)
    down_contributions_output = -tf.reduce_sum(down_contributions_output[:, :, :, :, 2],
                                               axis=-1) / tf.reduce_sum(
        down_contributions_output[:, :, :, :, 1], axis=-1)
    return periodic_post_process(grid_size, down_contributions_output, idx, inputs, left_contributions_output,
                                 right_contributions_output, up_contributions_output)


def black_box_dirichlet_output_transform(inputs, grid_size):
    idx = [i for i in range(0, grid_size - 1, 2)]
    up_contributions_output = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    up_contributions_output = tf.gather(up_contributions_output,
                                        [i for i in range(2, grid_size, 2)], axis=2)
    up_contributions_output = -tf.reduce_sum(up_contributions_output[:, :, :, :, 0],
                                             axis=-1) / tf.reduce_sum(
        up_contributions_output[:, :, :, :, 1], axis=-1)
    left_contributions_output = tf.gather(inputs, idx, axis=1)
    left_contributions_output = tf.gather(left_contributions_output,
                                          [i for i in range(1, grid_size, 2)], axis=2)
    left_contributions_output = -tf.reduce_sum(left_contributions_output[:, :, :, 2, :],
                                               axis=-1) / tf.reduce_sum(
        left_contributions_output[:, :, :, 1, :], axis=-1)
    right_contributions_output = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
    right_contributions_output = tf.gather(right_contributions_output,
                                           [i for i in range(1, grid_size, 2)], axis=2)
    right_contributions_output = -tf.reduce_sum(right_contributions_output[:, :, :, 0, :],
                                                axis=-1) / tf.reduce_sum(
        right_contributions_output[:, :, :, 1, :], axis=-1)

    down_contributions_output = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    down_contributions_output = tf.gather(down_contributions_output, idx, axis=2)
    down_contributions_output = -tf.reduce_sum(down_contributions_output[:, :, :, :, 2],
                                               axis=-1) / tf.reduce_sum(
        down_contributions_output[:, :, :, :, 1], axis=-1)
    return dirichlet_post_process(grid_size, down_contributions_output, idx, inputs, left_contributions_output,
                                  right_contributions_output, up_contributions_output)
