import tensorflow.compat.v1 as tf


# takes the partial outputs of either the transformed model output or the blackbox model output and rebuilds into a P matrix STENICL?
# TODO no idea

def periodic_post_process(grid_size, down_contributions_output, idx, inputs, left_contributions_output,
                          right_contributions_output, up_contributions_output):
    ones = tf.ones_like(down_contributions_output)
    # based on rule 2 given rule 1:
    up_right_contribution = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    up_right_contribution = tf.gather(up_right_contribution, [i for i in range(1, grid_size, 2)], axis=2)
    up_right_contribution = up_right_contribution[:, :, :, 0, 1]
    right_up_contribution = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    right_up_contribution = tf.gather(right_up_contribution, [i for i in range(1, grid_size, 2)], axis=2)
    right_up_contribution_additional_term = right_up_contribution[:, :, :, 0, 0]
    right_up_contribution = right_up_contribution[:, :, :, 1, 0]
    ru_center_ = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    ru_center_ = tf.gather(ru_center_, [i for i in range(1, grid_size, 2)], axis=2)
    ru_center_ = ru_center_[:, :, :, 1, 1]
    ru_contribution = -tf.expand_dims((right_up_contribution_additional_term +
                                       tf.multiply(right_up_contribution, right_contributions_output) +
                                       tf.multiply(up_right_contribution,
                                                   up_contributions_output)) / ru_center_, -1)
    up_left_contribution = tf.gather(inputs, idx, axis=1)
    up_left_contribution = tf.gather(up_left_contribution, [i for i in range(1, grid_size, 2)], axis=2)
    up_left_contribution = up_left_contribution[:, :, :, 2, 1]
    left_up_contribution = tf.gather(inputs, idx, axis=1)
    left_up_contribution = tf.gather(left_up_contribution, [i for i in range(1, grid_size, 2)], axis=2)
    left_up_contribution_additional_term = left_up_contribution[:, :, :, 2, 0]
    left_up_contribution = left_up_contribution[:, :, :, 1, 0]
    lu_center_ = tf.gather(inputs, idx, axis=1)
    lu_center_ = tf.gather(lu_center_, [i for i in range(1, grid_size, 2)], axis=2)
    lu_center_ = lu_center_[:, :, :, 1, 1]
    lu_contribution = -tf.expand_dims((left_up_contribution_additional_term +
                                       tf.multiply(up_left_contribution, up_contributions_output) +
                                       tf.multiply(left_up_contribution,
                                                   left_contributions_output)) / lu_center_, -1)
    down_left_contribution = tf.gather(inputs, idx, axis=1)
    down_left_contribution = tf.gather(down_left_contribution, idx, axis=2)
    down_left_contribution = down_left_contribution[:, :, :, 2, 1]
    left_down_contribution = tf.gather(inputs, idx, axis=1)
    left_down_contribution = tf.gather(left_down_contribution, idx, axis=2)
    left_down_contribution_additional_term = left_down_contribution[:, :, :, 2, 2]
    left_down_contribution = left_down_contribution[:, :, :, 1, 2]
    ld_center_ = tf.gather(inputs, idx, axis=1)
    ld_center_ = tf.gather(ld_center_, idx, axis=2)
    ld_center_ = ld_center_[:, :, :, 1, 1]
    ld_contribution = -tf.expand_dims((left_down_contribution_additional_term +
                                       tf.multiply(down_left_contribution, down_contributions_output) +
                                       tf.multiply(left_down_contribution,
                                                   left_contributions_output)) / ld_center_, -1)
    down_right_contribution = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    down_right_contribution = tf.gather(down_right_contribution, idx, axis=2)
    down_right_contribution = down_right_contribution[:, :, :, 0, 1]
    right_down_contribution = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    right_down_contribution = tf.gather(right_down_contribution, idx, axis=2)
    right_down_contribution_additional_term = right_down_contribution[:, :, :, 0, 2]
    right_down_contribution = right_down_contribution[:, :, :, 1, 2]
    rd_center_ = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
    rd_center_ = tf.gather(rd_center_, idx, axis=2)
    rd_center_ = rd_center_[:, :, :, 1, 1]
    rd_contribution = -tf.expand_dims((right_down_contribution_additional_term + tf.multiply(
        down_right_contribution, down_contributions_output) +
                                       tf.multiply(right_down_contribution,
                                                   right_contributions_output)) / rd_center_, -1)
    first_row = tf.concat([ld_contribution, tf.expand_dims(left_contributions_output, -1),
                           lu_contribution], -1)
    second_row = tf.concat([tf.expand_dims(down_contributions_output, -1),
                            tf.expand_dims(ones, -1), tf.expand_dims(up_contributions_output, -1)], -1)
    third_row = tf.concat([rd_contribution, tf.expand_dims(right_contributions_output, -1),
                           ru_contribution], -1)
    output = tf.stack([first_row, second_row, third_row], 0)
    output = tf.transpose(output, (1, 2, 3, 0, 4))
    return tf.to_complex128(output)


def dirichlet_post_process(grid_size, down_contributions_output, idx, inputs, left_contributions_output,
                           right_contributions_output, up_contributions_output):
    ones = tf.ones_like(down_contributions_output)

    # based on rule 2 given rule 1:
    # x,y = np.ix_([3, 1], [1, 3])
    up_right_contribution = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
    up_right_contribution = tf.gather(up_right_contribution, [i for i in range(2, grid_size, 2)], axis=2)
    up_right_contribution = up_right_contribution[:, :, :, 0, 1]
    right_up_contribution = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
    right_up_contribution = tf.gather(right_up_contribution, [i for i in range(2, grid_size, 2)], axis=2)
    right_up_contribution_additional_term = right_up_contribution[:, :, :, 0, 0]
    right_up_contribution = right_up_contribution[:, :, :, 1, 0]
    ru_center_ = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
    ru_center_ = tf.gather(ru_center_, [i for i in range(2, grid_size, 2)], axis=2)
    ru_center_ = ru_center_[:, :, :, 1, 1]
    ru_contribution = -tf.expand_dims((right_up_contribution_additional_term +
                                       tf.multiply(right_up_contribution, right_contributions_output) +
                                       tf.multiply(up_right_contribution,
                                                   up_contributions_output)) / ru_center_, -1)

    # x,y = np.ix_([3, 1], [3, 1])
    up_left_contribution = tf.gather(inputs, idx, axis=1)
    up_left_contribution = tf.gather(up_left_contribution, [i for i in range(2, grid_size, 2)], axis=2)
    up_left_contribution = up_left_contribution[:, :, :, 2, 1]
    left_up_contribution = tf.gather(inputs, idx, axis=1)
    left_up_contribution = tf.gather(left_up_contribution, [i for i in range(2, grid_size, 2)], axis=2)
    left_up_contribution_additional_term = left_up_contribution[:, :, :, 2, 0]
    left_up_contribution = left_up_contribution[:, :, :, 1, 0]
    lu_center_ = tf.gather(inputs, idx, axis=1)
    lu_center_ = tf.gather(lu_center_, [i for i in range(2, grid_size, 2)], axis=2)
    lu_center_ = lu_center_[:, :, :, 1, 1]
    lu_contribution = -tf.expand_dims((left_up_contribution_additional_term +
                                       tf.multiply(up_left_contribution, up_contributions_output) +
                                       tf.multiply(left_up_contribution,
                                                   left_contributions_output)) / lu_center_, -1)

    # x,y = np.ix_([1, 3], [3, 1])
    down_left_contribution = tf.gather(inputs, idx, axis=1)
    down_left_contribution = tf.gather(down_left_contribution, idx, axis=2)
    down_left_contribution = down_left_contribution[:, :, :, 2, 1]
    left_down_contribution = tf.gather(inputs, idx, axis=1)
    left_down_contribution = tf.gather(left_down_contribution, idx, axis=2)
    left_down_contribution_additional_term = left_down_contribution[:, :, :, 2, 2]
    left_down_contribution = left_down_contribution[:, :, :, 1, 2]
    ld_center_ = tf.gather(inputs, idx, axis=1)
    ld_center_ = tf.gather(ld_center_, idx, axis=2)
    ld_center_ = ld_center_[:, :, :, 1, 1]
    ld_contribution = -tf.expand_dims((left_down_contribution_additional_term +
                                       tf.multiply(down_left_contribution, down_contributions_output) +
                                       tf.multiply(left_down_contribution,
                                                   left_contributions_output)) / ld_center_, -1)

    # x,y = np.ix_([1, 3], [1, 3])
    down_right_contribution = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
    down_right_contribution = tf.gather(down_right_contribution, idx, axis=2)
    down_right_contribution = down_right_contribution[:, :, :, 0, 1]
    right_down_contribution = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
    right_down_contribution = tf.gather(right_down_contribution, idx, axis=2)
    right_down_contribution_additional_term = right_down_contribution[:, :, :, 0, 2]
    right_down_contribution = right_down_contribution[:, :, :, 1, 2]
    rd_center_ = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
    rd_center_ = tf.gather(rd_center_, idx, axis=2)
    rd_center_ = rd_center_[:, :, :, 1, 1]
    rd_contribution = -tf.expand_dims((right_down_contribution_additional_term + tf.multiply(
        down_right_contribution, down_contributions_output) +
                                       tf.multiply(right_down_contribution,
                                                   right_contributions_output)) / rd_center_, -1)

    first_row = tf.concat([ld_contribution, tf.expand_dims(left_contributions_output, -1),
                           lu_contribution], -1)
    second_row = tf.concat([tf.expand_dims(down_contributions_output, -1),
                            tf.expand_dims(ones, -1), tf.expand_dims(up_contributions_output, -1)], -1)
    third_row = tf.concat([rd_contribution, tf.expand_dims(right_contributions_output, -1),
                           ru_contribution], -1)

    output = tf.stack([first_row, second_row, third_row], 0)
    output = tf.transpose(output, (1, 2, 3, 0, 4))
    return tf.to_complex128(output)
