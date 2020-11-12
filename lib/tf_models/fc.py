import tensorflow as tf

keras = tf.keras
layers = keras.layers


class FC(keras.Model):

    def __init__(self, hidden_layers, fc_activation='relu'):
        super(FC, self).__init__()

        self.__fc_layers = []
        for unit in hidden_layers:
            self.__fc_layers.append(layers.Dense(unit, activation=fc_activation))

        self.__softmax = layers.Dense(2, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        embedding_1, embedding_2 = inputs

        x = tf.concat([embedding_1, embedding_2, embedding_1 - embedding_2, embedding_1 + embedding_2], axis=-1)

        for fc_layer in self.__fc_layers:
            x = fc_layer(x)

        return self.__softmax(x)
