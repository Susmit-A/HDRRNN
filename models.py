from cells import *
from layers import *


class RNNModel(tf.keras.Model):
    def __init__(self, input_channels, starting_channels, out_channels, state_channels, cell, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_channels = state_channels
        self._name = name

        inp = Input((None, None, input_channels * 2))
        x = Conv2D(starting_channels, (3, 3), padding='same', activation=LeakyReLU())(inp)
        x = Conv2D(starting_channels, (3, 3), padding='same', activation=LeakyReLU())(x)
        x = Conv2D(starting_channels, (3, 3), padding='same', activation=LeakyReLU())(x)
        x = Conv2D(state_channels, (3, 3), padding='same', activation=LeakyReLU())(x)
        self.encoder = Model(inputs=[inp], outputs=[x])

        inp = Input((None, None, state_channels))
        x = SDC(starting_channels, (3, 3))(inp)
        x = SDC(starting_channels, (3, 3))(x)
        x = Conv2D(out_channels, (3, 3), padding='same', activation=None)(x)
        x = Activation('sigmoid', dtype=tf.float32)(x)
        self.decoder = Model(inputs=[inp], outputs=[x])

        inp = Input((None, None, state_channels))
        state = Input((None, None, state_channels))
        if cell == 'lstm':
            x, s = ConvLSTMCell(state_channels)([inp, [inp, state]])
            self.recurrent = Model(inputs=[inp, state], outputs=[x, s[1]])
        elif cell == 'gru':
            x, s = CustomGRUCell(state_channels)([inp, state])
            self.recurrent = Model(inputs=[inp, state], outputs=[x, s])
        elif cell == 'sgm':
            x, s = SGMCell(state_channels)([inp, [inp, state]])
            self.recurrent = Model(inputs=[inp, state], outputs=[x, s[1]])
        else:
            raise ValueError("Cell should be one of 'lstm', 'sgm', 'gru'")

    @property
    def name(self):
        return self._name

    def call(self, inputs, **kwargs):
        ref = inputs[1]
        inputs = inputs[0]
        unstacked = tf.unstack(inputs, axis=1)
        state = tf.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3], self.state_channels), dtype=tf.float32)
        for i in range(len(unstacked)):
            x = tf.concat([unstacked[i], ref], axis=-1)
            x = self.encoder(x)
            x, state = self.recurrent([x, state])

        out = self.decoder(x)
        return out


class BiRNNModel(tf.keras.Model):
    def __init__(self, input_channels, starting_channels, out_channels, state_channels, cell, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_channels = state_channels
        self._name = name

        inp = Input((None, None, input_channels * 2))
        x = Conv2D(starting_channels, (3, 3), padding='same', activation=LeakyReLU())(inp)
        x = Conv2D(starting_channels, (3, 3), padding='same', activation=LeakyReLU())(x)
        x = Conv2D(starting_channels, (3, 3), padding='same', activation=LeakyReLU())(x)
        x = Conv2D(state_channels, (3, 3), padding='same', activation=LeakyReLU())(x)
        self.encoder = Model(inputs=[inp], outputs=[x])

        inp = Input((None, None, state_channels * 2))
        x = SDC(starting_channels, (3, 3))(inp)
        x = SDC(starting_channels, (3, 3))(x)
        x = Conv2D(out_channels, (3, 3), padding='same', activation=None)(x)
        x = Activation('sigmoid', dtype=tf.float32)(x)
        self.decoder = Model(inputs=[inp], outputs=[x])

        inp = Input((None, None, state_channels))
        state = Input((None, None, state_channels))
        inp_reverse = Input((None, None, state_channels))
        state_reverse = Input((None, None, state_channels))

        if cell == 'lstm':
            x, s = ConvLSTMCell(state_channels)([inp, [inp, state]])
            self.recurrent = Model(inputs=[inp, state], outputs=[x, s[1]])
            x, s = ConvLSTMCell(state_channels)([inp_reverse, [inp_reverse, state_reverse]])
            self.recurrent_reverse = Model(inputs=[inp_reverse, state_reverse], outputs=[x, s[1]])
        elif cell == 'gru':
            x, s = CustomGRUCell(state_channels)([inp, state])
            self.recurrent = Model(inputs=[inp, state], outputs=[x, s])
            x, s = CustomGRUCell(state_channels)([inp_reverse, state_reverse])
            self.recurrent_reverse = Model(inputs=[inp_reverse, state_reverse], outputs=[x, s])
        elif cell == 'sgm':
            x, s = SGMCell(state_channels)([inp, [inp, state]])
            self.recurrent = Model(inputs=[inp, state], outputs=[x, s[1]])
            x, s = SGMCell(state_channels)([inp_reverse, [inp_reverse, state_reverse]])
            self.recurrent_reverse = Model(inputs=[inp_reverse, state_reverse], outputs=[x, s[1]])
        else:
            raise ValueError("Cell should be one of 'lstm', 'sgm', 'gru'")

    @property
    def name(self):
        return self._name

    def call(self, inputs, **kwargs):
        ref = inputs[1]
        inputs = inputs[0]
        unstacked = tf.unstack(inputs, axis=1)
        state = state_rev = tf.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3], self.state_channels), dtype=tf.float32)
        for i in range(len(unstacked)):
            x = tf.concat([unstacked[i], ref], axis=-1)
            x = self.encoder(x)
            x, state = self.recurrent([x, state])

            x_rev = tf.concat([unstacked[-(i+1)], ref], axis=-1)
            x_rev = self.encoder(x_rev)
            x_rev, state_rev = self.recurrent_reverse([x_rev, state_rev])

        x = tf.concat([x, x_rev], axis=-1)
        out = self.decoder(x)
        return out

models = {
    'unidirectional': RNNModel,
    'bidirectional': BiRNNModel
}
