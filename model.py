import numpy as np
import tensorflow.compat.v1 as tf
from postprocess import periodic_post_process, dirichlet_post_process


@tf.function
def dirichlet_input_transform(inputs: tf.Tensor, grid_size: int):
    coarse_grid_size = grid_size // 2
    batch_size = inputs.shape[0]
    right_contributions_input = tf.gather(params=inputs,
                                          indices=[i for i in range(2, grid_size, 2)], axis=1)
    right_contributions_input = tf.gather(params=right_contributions_input,
                                          indices=[i for i in range(1, grid_size, 2)], axis=2)
    idx = [i for i in range(0, grid_size - 1, 2)]
    left_contributions_input = tf.gather(params=inputs, indices=idx, axis=1)
    left_contributions_input = tf.gather(params=left_contributions_input,
                                         indices=[i for i in range(1, grid_size, 2)], axis=2)
    left_contributions_input = tf.reshape(tensor=left_contributions_input,
                                          shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))

    up_contributions_input = tf.gather(params=inputs, indices=[i for i in range(1, grid_size, 2)], axis=1)
    up_contributions_input = tf.gather(params=up_contributions_input,
                                       indices=[i for i in range(2, grid_size, 2)], axis=2)
    up_contributions_input = tf.reshape(tensor=up_contributions_input,
                                        shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))

    down_contributions_input = tf.gather(params=inputs,
                                         indices=[i for i in range(1, grid_size, 2)], axis=1)
    down_contributions_input = tf.gather(params=down_contributions_input, indices=idx, axis=2)
    down_contributions_input = tf.reshape(tensor=down_contributions_input,
                                          shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))
    center_contributions_input = tf.gather(params=inputs,
                                           indices=[i for i in range(1, grid_size, 2)], axis=1)
    center_contributions_input = tf.gather(params=center_contributions_input,
                                           indices=[i for i in range(1, grid_size, 2)],
                                           axis=2)
    center_contributions_input = tf.reshape(tensor=center_contributions_input,
                                            shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))

    inputs_combined = tf.concat([right_contributions_input, left_contributions_input,
                                 up_contributions_input, down_contributions_input,
                                 center_contributions_input], 0)

    flattened = tf.reshape(inputs_combined, (-1, 9))
    temp = coarse_grid_size ** 2

    flattened = tf.concat([flattened[:batch_size * temp],
                           flattened[temp * batch_size:temp * 2 * batch_size],
                           flattened[temp * 2 * batch_size:temp * 3 * batch_size],
                           flattened[temp * 3 * batch_size:temp * 4 * batch_size],
                           flattened[temp * 4 * batch_size:]], -1)

    return flattened


@tf.function
def periodic_input_transform(inputs: tf.Tensor, grid_size: int) -> tf.Tensor:
    coarse_grid_size = grid_size // 2
    batch_size = inputs.shape[0]
    right_contributions_input = tf.gather(params=inputs,
                                          indices=[i for i in range(1, grid_size, 2)], axis=1)
    right_contributions_input = tf.gather(params=right_contributions_input,
                                          indices=[i for i in range(0, grid_size, 2)], axis=2)
    idx = [(i - 1) % grid_size for i in range(0, grid_size, 2)]
    left_contributions_input = tf.gather(params=inputs, indices=idx, axis=1)
    left_contributions_input = tf.gather(params=left_contributions_input,
                                         indices=[i for i in range(0, grid_size, 2)], axis=2)
    left_contributions_input = tf.reshape(tensor=left_contributions_input,
                                          shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))

    up_contributions_input = tf.gather(params=inputs, indices=[i for i in range(0, grid_size, 2)],
                                       axis=1)
    up_contributions_input = tf.gather(params=up_contributions_input,
                                       indices=[i for i in range(1, grid_size, 2)], axis=2)
    up_contributions_input = tf.reshape(tensor=up_contributions_input,
                                        shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))

    down_contributions_input = tf.gather(params=inputs,
                                         indices=[i for i in range(0, grid_size, 2)], axis=1)
    down_contributions_input = tf.gather(params=down_contributions_input, indices=idx, axis=2)
    down_contributions_input = tf.reshape(tensor=down_contributions_input,
                                          shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))
    center_contributions_input = tf.gather(params=inputs,
                                           indices=[i for i in range(0, grid_size, 2)], axis=1)
    center_contributions_input = tf.gather(params=center_contributions_input,
                                           indices=[i for i in range(0, grid_size, 2)], axis=2)
    center_contributions_input = tf.reshape(tensor=center_contributions_input,
                                            shape=(-1, coarse_grid_size, coarse_grid_size, 3, 3))

    inputs_combined = tf.concat([right_contributions_input, left_contributions_input,
                                 up_contributions_input, down_contributions_input,
                                 center_contributions_input], 0)

    flattened = tf.reshape(inputs_combined, (-1, 9))

    temp = (grid_size // 2) ** 2

    flattened = tf.concat([flattened[:batch_size * temp],
                           flattened[temp * batch_size:temp * 2 * batch_size],
                           flattened[temp * 2 * batch_size:temp * 3 * batch_size],
                           flattened[temp * 3 * batch_size:temp * 4 * batch_size],
                           flattened[temp * 4 * batch_size:]], -1)
    return flattened


@tf.function
def periodic_output_transform(inputs: tf.Tensor, grid_size, model_output, index=None, pos=-1.,
                              phase='Training') -> tf.Tensor:
    idx = [(i - 1) % grid_size for i in range(0, grid_size, 2)]
    coarse_grid_size = grid_size // 2
    if index is not None:
        indices = tf.constant([[index]])
        updates = [tf.to_double(pos)]
        shape = tf.constant([2 * 2 * 2 * 8])
        scatter = tf.scatter_nd(indices, updates, shape)
        x = model_output + tf.reshape(scatter, (-1, 2, 2, 8))
        ld_contribution = x[:, :, :, 0]
        left_contributions_output = x[:, :, :, 1]
        lu_contribution = x[:, :, :, 2]
        down_contributions_output = x[:, :, :, 3]
        up_contributions_output = x[:, :, :, 4]
        ones = tf.ones_like(up_contributions_output)
        right_contributions_output = x[:, :, :, 6]
        rd_contribution = x[:, :, :, 5]
        ru_contribution = x[:, :, :, 7]
        first_row = tf.concat(
            [tf.expand_dims(ld_contribution, -1), tf.expand_dims(left_contributions_output, -1),
             tf.expand_dims(lu_contribution, -1)], -1)
        second_row = tf.concat([tf.expand_dims(down_contributions_output, -1),
                                tf.expand_dims(ones, -1), tf.expand_dims(up_contributions_output, -1)], -1)
        third_row = tf.concat(
            [tf.expand_dims(rd_contribution, -1), tf.expand_dims(right_contributions_output, -1),
             tf.expand_dims(ru_contribution, -1)], -1)

        output = tf.stack([first_row, second_row, third_row], 0)
        output = tf.transpose(output, (1, 2, 3, 0, 4))
        return tf.to_complex128(output)
    else:
        x = tf.reshape(model_output, (-1, coarse_grid_size, coarse_grid_size, 4))
        jm1 = [(i - 1) % coarse_grid_size for i in range(coarse_grid_size)]
        jp1 = [(i + 1) % coarse_grid_size for i in range(coarse_grid_size)]
        right_contributions_output = x[:, :, :, 0] / (tf.gather(x[:, :, :, 1], jp1, axis=1) + x[:, :, :, 0])
        left_contributions_output = x[:, :, :, 1] / (x[:, :, :, 1] + tf.gather(x[:, :, :, 0], jm1, axis=1))
        up_contributions_output = x[:, :, :, 2] / (x[:, :, :, 2] + tf.gather(x[:, :, :, 3], jp1, axis=2))
        down_contributions_output = x[:, :, :, 3] / (tf.gather(x[:, :, :, 2], jm1, axis=2) + x[:, :, :, 3])

    return periodic_post_process(grid_size, down_contributions_output, idx, inputs, left_contributions_output,
                                 right_contributions_output, up_contributions_output)


@tf.function
def dirichlet_output_tranform(inputs, grid_size, model_output, index=None, pos=-1., phase='Training'):
    idx = [i for i in range(0, grid_size - 1, 2)]
    coarse_grid_size = grid_size // 2
    if index is not None:
        indices = tf.constant([[index]])
        updates = [tf.to_double(pos)]
        shape = tf.constant([2 * 2 * 2 * 8])
        scatter = tf.scatter_nd(indices, updates, shape)
        x = model_output + tf.reshape(scatter, (-1, 2, 2, 8))
        ld_contribution = x[:, :, :, 0]
        left_contributions_output = x[:, :, :, 1]
        lu_contribution = x[:, :, :, 2]
        down_contributions_output = x[:, :, :, 3]
        up_contributions_output = x[:, :, :, 4]
        ones = tf.ones_like(up_contributions_output)
        right_contributions_output = x[:, :, :, 6]
        rd_contribution = x[:, :, :, 5]
        ru_contribution = x[:, :, :, 7]
        first_row = tf.concat(
            [tf.expand_dims(ld_contribution, -1), tf.expand_dims(left_contributions_output, -1),
             tf.expand_dims(lu_contribution, -1)], -1)
        second_row = tf.concat([tf.expand_dims(down_contributions_output, -1),
                                tf.expand_dims(ones, -1), tf.expand_dims(up_contributions_output, -1)], -1)
        third_row = tf.concat(
            [tf.expand_dims(rd_contribution, -1), tf.expand_dims(right_contributions_output, -1),
             tf.expand_dims(ru_contribution, -1)], -1)

        output = tf.stack([first_row, second_row, third_row], 0)
        output = tf.transpose(output, (1, 2, 3, 0, 4))
        return tf.to_complex128(output)
    else:
        x = tf.reshape(model_output, (-1, coarse_grid_size, coarse_grid_size, 4))
        jm1 = [(i - 0) % coarse_grid_size for i in range(coarse_grid_size - 1)]
        jp1 = [(i + 1) % coarse_grid_size for i in range(coarse_grid_size - 1)]
        right_contributions_output = x[:, :-1, :, 0] / (tf.gather(x[:, :, :, 1], jp1, axis=1) + x[:, :-1, :, 0])
        left_contributions_output = x[:, 1:, :, 1] / (x[:, 1:, :, 1] + tf.gather(x[:, :, :, 0], jm1, axis=1))
        up_contributions_output = x[:, :, :-1, 2] / (x[:, :, :-1, 2] + tf.gather(x[:, :, :, 3], jp1, axis=2))
        down_contributions_output = x[:, :, 1:, 3] / (tf.gather(x[:, :, :, 2], jm1, axis=2) + x[:, :, 1:, 3])

        # complete right with black box:
        right_contributions_output_bb = tf.gather(inputs, [i for i in range(2, grid_size, 2)], axis=1)
        right_contributions_output_bb = tf.gather(right_contributions_output_bb,
                                                  [i for i in range(1, grid_size, 2)], axis=2)
        right_contributions_output_bb = -tf.reduce_sum(right_contributions_output_bb[:, :, :, 0, :],
                                                       axis=-1) / tf.reduce_sum(
            right_contributions_output_bb[:, :, :, 1, :], axis=-1)
        right_contributions_output_bb = tf.reshape(right_contributions_output_bb[:, -1, :], (1, 1, -1))
        right_contributions_output = tf.concat([right_contributions_output, right_contributions_output_bb],
                                               axis=1)
        left_contributions_output_bb = tf.gather(inputs, idx, axis=1)
        left_contributions_output_bb = tf.gather(left_contributions_output_bb,
                                                 [i for i in range(1, grid_size, 2)], axis=2)
        left_contributions_output_bb = -tf.reduce_sum(left_contributions_output_bb[:, :, :, 2, :],
                                                      axis=-1) / tf.reduce_sum(
            left_contributions_output_bb[:, :, :, 1, :], axis=-1)
        left_contributions_output_bb = tf.reshape(left_contributions_output_bb[:, 0, :], (1, 1, -1))
        left_contributions_output = tf.concat([left_contributions_output_bb, left_contributions_output], axis=1)
        up_contributions_output_bb = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
        up_contributions_output_bb = tf.gather(up_contributions_output_bb,
                                               [i for i in range(2, grid_size, 2)], axis=2)
        up_contributions_output_bb = -tf.reduce_sum(up_contributions_output_bb[:, :, :, :, 0],
                                                    axis=-1) / tf.reduce_sum(
            up_contributions_output_bb[:, :, :, :, 1], axis=-1)
        up_contributions_output_bb = tf.reshape(up_contributions_output_bb[:, :, -1], (1, -1, 1))
        up_contributions_output = tf.concat([up_contributions_output, up_contributions_output_bb], axis=-1)
        down_contributions_output_bb = tf.gather(inputs, [i for i in range(1, grid_size, 2)], axis=1)
        down_contributions_output_bb = tf.gather(down_contributions_output_bb, idx, axis=2)
        down_contributions_output_bb = -tf.reduce_sum(down_contributions_output_bb[:, :, :, :, 2],
                                                      axis=-1) / tf.reduce_sum(
            down_contributions_output_bb[:, :, :, :, 1], axis=-1)
        down_contributions_output_bb = tf.reshape(down_contributions_output_bb[:, :, 0], (1, -1, 1))
        down_contributions_output = tf.concat([down_contributions_output_bb, down_contributions_output],
                                              axis=-1)
    return dirichlet_post_process(grid_size, down_contributions_output, idx, inputs, left_contributions_output,
                                  right_contributions_output, up_contributions_output)


class PNetwork(tf.keras.Model):
    def __init__(self):
        super(PNetwork, self).__init__()
        width = 100
        self.linear0 = tf.keras.layers.Dense(width, kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                             use_bias=False)
        self.num_layers = 100
        for i in range(1, self.num_layers):
            setattr(self, "linear%i" % i, tf.keras.layers.Dense(width, use_bias=False,
                                                                kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                                                kernel_initializer=tf.initializers.truncated_normal(
                                                                    stddev=i ** (-1 / 2) * np.sqrt(2. / width))))
            setattr(self, "bias_1%i" % i, tf.Variable([0.], dtype=tf.float64))
            setattr(self, "linear%i" % (i + 1), tf.keras.layers.Dense(width, use_bias=False,
                                                                      kernel_regularizer=tf.keras.regularizers.l2(
                                                                          1e-7),
                                                                      kernel_initializer=tf.zeros_initializer()))  # surely this lineari+1 just gets overwritten at next loop?
            setattr(self, "bias_2%i" % i, tf.Variable([0.], dtype=tf.float64))
            setattr(self, "bias_3%i" % i, tf.Variable([0.], dtype=tf.float64))
            setattr(self, "bias_4%i" % i, tf.Variable([0.], dtype=tf.float64))
            setattr(self, "multiplier_%i" % i, tf.Variable([1.], dtype=tf.float64))

        self.output_layer = tf.keras.layers.Dense(4, use_bias=True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.new_output = tf.Variable(0.5 * tf.random_normal(shape=[2 * 2 * 2 * 8], dtype=tf.float64),
                                      dtype=tf.float64)

    def call(self, inputs, training=None, mask=None):
        x = self.linear0(inputs)
        x = tf.nn.relu(x)
        for i in range(1, self.num_layers, 2):
            x1 = getattr(self, "bias_1%i" % i) + x
            x1 = getattr(self, "linear%i" % i)(x1)
            x1 = x1 + getattr(self, "bias_2%i" % i) + x1
            x1 = tf.nn.relu(x1)
            x1 = x1 + getattr(self, "bias_3%i" % i) + x1
            x1 = getattr(self, "linear%i" % (i + 1))(x1)
            x1 = tf.multiply(x1, getattr(self, "multiplier_%i" % i))
            x = x + x1
            x = x + getattr(self, "bias_4%i" % i)
            x = tf.nn.relu(x)

        x = self.output_layer(x)
        return x


class PNetworkSimple(tf.keras.Model):
    def __init__(self):
        super(PNetworkSimple, self).__init__()

        width = 100
        self.num_layers = 4
        setattr(self, "dense0", tf.keras.layers.Dense(units=100, use_bias=True,
                                                      activation='relu',
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                                      kernel_initializer='glorot_uniform',
                                                      bias_initializer='zeros'))
        setattr(self, "dense1", tf.keras.layers.Dense(units=50, use_bias=True,
                                                      activation='relu',
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                                      kernel_initializer='glorot_uniform',
                                                      bias_initializer='zeros'))
        setattr(self, "dense2", tf.keras.layers.Dense(units=30, use_bias=True,
                                                      activation='relu',
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                                      kernel_initializer='glorot_uniform',
                                                      bias_initializer='zeros'))
        setattr(self, "dense3", tf.keras.layers.Dense(units=50, use_bias=True,
                                                      activation='relu',
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-7),
                                                      kernel_initializer='glorot_uniform',
                                                      bias_initializer='zeros'))
        self.output_layer = tf.keras.layers.Dense(4, use_bias=True,
                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.new_output = tf.Variable(0.5 * tf.random_normal(shape=[2 * 2 * 2 * 8], dtype=tf.float64),
                                      dtype=tf.float64)

    def call(self, inputs, training=None, mask=None):
        x = self.dense0(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.output_layer(x)
        return x
