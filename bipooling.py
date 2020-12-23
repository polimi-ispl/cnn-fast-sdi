from keras.layers import Layer
import tensorflow as tf


class PairwiseCorrPooling(Layer):

    def __init__(self, pooling=True, **kwargs):
        self.pooling = pooling
        super(PairwiseCorrPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PairwiseCorrPooling, self).build(input_shape)

    def call(self, x):
        halfdim3 = x.get_shape()[-1] // 2
        print(x.get_shape(), x.dtype)
        y = x[:, :, :, :halfdim3] * x[:, :, :, halfdim3:]
        if self.pooling:
            y = tf.reduce_mean(y, [1, 2], keep_dims=True)
        print(y.get_shape(), y.dtype)
        return y

    def compute_output_shape(self, input_shape):
        if self.pooling:
            return (input_shape[0], 1, 1, input_shape[3] // 2)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // 2)
