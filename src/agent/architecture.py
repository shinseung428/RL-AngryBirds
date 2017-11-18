import tensorflow as tf

def conv2d(x, input_filters, output_filters, 
		   kernel, strides, padding = 1, 
		   mode='CONSTANT', name='conv'):
    with tf.variable_scope(name) as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name=name)

        return conv

def leaky_relu(input, val):
    with tf.name_scope("leaky"):
        x = tf.identity(input)
        return (0.5 * (1 + val)) * x + (0.5 * (1 - val)) * tf.abs(x)

def batchnorm(input, name="batchnorm"):
    with tf.variable_scope(name):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized