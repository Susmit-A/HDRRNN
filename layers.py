from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow.keras.layers as layers


class DRIB_Unit(Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding='same', *args, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        super().__init__(*args, **kwargs)

    def dilated_conv(self, dilation_rate):
        return Conv2D(
            self.filters,
            self.kernel_size,
            dilation_rate=dilation_rate,
            activation=LeakyReLU(0.2),
            padding=self.padding,
            use_bias=True,
            kernel_initializer='he_normal'
        )

    def build(self, input_shape):
        self.dilated_conv1 = self.dilated_conv(dilation_rate=(1, 1))
        self.dilated_conv2 = self.dilated_conv(dilation_rate=(1, 1))
        self.dilated_conv3 = self.dilated_conv(dilation_rate=(1, 1))
        self.concat = Concatenate()
        self.dilated_conv4 = Conv2D(self.filters, (3, 3), activation=LeakyReLU(0.2), dilation_rate=(1, 1), padding='same')

    def call(self, inputs, **kwargs):
        x = inputs
        x1 = self.dilated_conv1(x)
        x2 = tf.nn.conv2d(x, self.dilated_conv1.weights[0], strides=(1, 1), padding='SAME', dilations=(2, 2))
        x3 = tf.nn.conv2d(x, self.dilated_conv1.weights[0], strides=(1, 1), padding='SAME', dilations=(4, 4))

        x1 = tf.nn.leaky_relu(x1, 0.2)
        x2 = tf.nn.leaky_relu(x2, 0.2)
        x3 = tf.nn.leaky_relu(x3, 0.2)

        x1 = self.dilated_conv2(x1)
        x2 = tf.nn.conv2d(x2, self.dilated_conv2.weights[0], strides=(1, 1), padding='SAME', dilations=(2, 2))
        x3 = tf.nn.conv2d(x3, self.dilated_conv2.weights[0], strides=(1, 1), padding='SAME', dilations=(4, 4))

        x1 = tf.nn.leaky_relu(x1, 0.2)
        x2 = tf.nn.leaky_relu(x2, 0.2)
        x3 = tf.nn.leaky_relu(x3, 0.2)

        x1 = self.dilated_conv3(x1)
        x2 = tf.nn.conv2d(x2, self.dilated_conv3.weights[0], strides=(1, 1), padding='SAME', dilations=(2, 2))
        x3 = tf.nn.conv2d(x3, self.dilated_conv3.weights[0], strides=(1, 1), padding='SAME', dilations=(4, 4))

        x1 = tf.nn.leaky_relu(x1, 0.2)
        x2 = tf.nn.leaky_relu(x2, 0.2)
        x3 = tf.nn.leaky_relu(x3, 0.2)

        x = self.concat([x1, x2, x3])
        x = self.dilated_conv4(x)
        return x


class DRIB_Unshared(Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding='same', *args, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        super().__init__(*args, **kwargs)

    def dilated_conv(self, dilation_rate):
        return Conv2D(
            self.filters,
            self.kernel_size,
            dilation_rate=dilation_rate,
            activation=None,
            padding=self.padding,
            use_bias=False
        )

    def build(self, input_shape):
        self.dilated_conv1_1 = self.dilated_conv(dilation_rate=(1, 1))
        self.dilated_conv1_2 = self.dilated_conv(dilation_rate=(2, 2))
        self.dilated_conv1_3 = self.dilated_conv(dilation_rate=(4, 4))

        self.dilated_conv2_1 = self.dilated_conv(dilation_rate=(1, 1))
        self.dilated_conv2_2 = self.dilated_conv(dilation_rate=(2, 2))
        self.dilated_conv2_3 = self.dilated_conv(dilation_rate=(4, 4))

        self.dilated_conv3_1 = self.dilated_conv(dilation_rate=(1, 1))
        self.dilated_conv3_2 = self.dilated_conv(dilation_rate=(2, 2))
        self.dilated_conv3_3 = self.dilated_conv(dilation_rate=(4, 4))
        self.concat = Concatenate()
        self.dilated_conv4 = Conv2D(self.filters, (1, 1), activation=LeakyReLU(0.2), dilation_rate=(1, 1), kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        x = inputs
        x1 = self.dilated_conv1_1(x)
        x2 = self.dilated_conv1_2(x)
        x3 = self.dilated_conv1_3(x)

        x1 = self.dilated_conv2_1(x1)
        x2 = self.dilated_conv2_2(x2)
        x3 = self.dilated_conv2_3(x3)

        x1 = self.dilated_conv3_1(x1)
        x2 = self.dilated_conv3_2(x2)
        x3 = self.dilated_conv3_3(x3)

        x = self.concat([x1, x2, x3])
        x = self.dilated_conv4(x)
        return x


class SDC(Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding='same', *args, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, self.kernel_size, padding='same', activation=LeakyReLU())
        self.conv2 = Conv2D(self.filters, self.kernel_size, padding='same', activation=LeakyReLU(), dilation_rate=(2, 2))
        self.conv3 = Conv2D(self.filters, self.kernel_size, padding='same', activation=LeakyReLU(), dilation_rate=(3, 3))
        self.concat = Concatenate()

    def call(self, inputs, **kwargs):
        x = inputs
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = self.concat([x1, x2, x3])
        return x


class PSwish(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.wt = self.add_weight('weight', initializer='ones', trainable=True, constraint=tf.keras.constraints.non_neg())

    def call(self, inputs, **kwargs):
        x = inputs
        return x * tf.nn.sigmoid(x * self.wt) * tf.math.tanh(x)
