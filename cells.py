from tensorflow.keras.layers import *
from layers import *


class ConvLSTMCell(Layer):
    def __init__(self, state_channels=32, kernel_size=(3, 3), **kwargs):
        self.state_channels = state_channels
        self.ks = kernel_size

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.forget_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='sigmoid')
        self.candidate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='tanh')
        self.input_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='sigmoid')
        self.input_output_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='sigmoid')
        self.state_output_gate_conv = Activation(activation='tanh')

    def call(self, inputs, **kwargs):
        """
        :param inputs: [input, previous_state]
        :return: [output, next_state]
        """
        prev_output, prev_state = inputs[1]
        inputs = inputs[0]
        concat = tf.concat([inputs, prev_output], axis=-1)

        forget_out = self.forget_gate_conv(concat)

        input_gate_out = self.input_gate_conv(concat)
        candidate_out = self.candidate_conv(concat)
        input_gate_out = tf.multiply(input_gate_out, candidate_out)

        input_output_gate_out = self.input_output_gate_conv(concat)

        new_state = tf.multiply(prev_state, forget_out)
        new_state = new_state + input_gate_out

        state_output_gate_out = self.state_output_gate_conv(new_state)

        new_output = tf.multiply(state_output_gate_out, input_output_gate_out)

        return new_output, [new_output, new_state]


class CustomGRUCell(Layer):
    def __init__(self, state_channels=32, kernel_size=(3, 3), **kwargs):
        self.state_channels = state_channels
        self.ks = kernel_size

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.reset_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='sigmoid')
        self.update_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='sigmoid')
        self.input_output_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='tanh')
        self.ones = tf.ones((1, 1, 1, self.state_channels), dtype=tf.keras.mixed_precision.experimental.global_policy().compute_dtype)

    def call(self, inputs, **kwargs):
        """
        :param inputs: [input, previous_state]
        :return: [output, next_state]
        """
        prev_output = inputs[1]
        inputs = inputs[0]
        concat = tf.concat([inputs, prev_output], axis=-1)

        reset_conv_out = self.reset_gate_conv(concat)
        reset_gate_out = reset_conv_out * prev_output

        update_conv_out = self.update_gate_conv(concat)
        update_gate_out = prev_output * (self.ones - update_conv_out)

        output_gate_input = tf.concat([reset_gate_out, inputs], axis=-1)
        output_conv_out = self.input_output_gate_conv(output_gate_input)
        output = output_conv_out * update_conv_out
        output = output + update_gate_out

        return output, output


class SGMCell(Layer):
    def __init__(self, state_channels=32, kernel_size=(3, 3), **kwargs):
        self.state_channels = state_channels
        self.ks = kernel_size

        super().__init__(**kwargs)

    def swish(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)

    def build(self, input_shape):
        self.transform_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation=self.swish)
        self.candidate_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation=self.swish)
        self.update_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation='sigmoid')
        self.output_gate_conv = Conv2D(self.state_channels, self.ks, padding='same', activation=self.swish)
        self.ones = tf.ones((1, 1, 1, self.state_channels), dtype=tf.keras.mixed_precision.experimental.global_policy().compute_dtype)

    def call(self, inputs, **kwargs):
        """
        :param inputs: [input, previous_state]
        :return: [output, next_state]
        """
        prev_output, prev_state = inputs[1]
        inputs = inputs[0]
        concat = tf.concat([inputs, prev_output], axis=-1)

        candidate_conv_out = self.candidate_gate_conv(concat)
        transform_conv_out = self.transform_gate_conv(prev_state)

        update_gate_out = self.update_gate_conv(tf.concat([candidate_conv_out, transform_conv_out], axis=-1))
        new_state = (update_gate_out * transform_conv_out) + ((self.ones - update_gate_out) * candidate_conv_out)

        output_gate_input = tf.concat([new_state, prev_state + inputs], axis=-1)
        output = self.output_gate_conv(output_gate_input)

        return output, [output, new_state]
