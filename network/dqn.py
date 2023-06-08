import tensorflow as tf


class DQN(tf.keras.Model):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()

        # input layer
        # self.input_layer = tf.keras.layers.Dense(128, activation='relu')
        self.input_layer = tf.keras.layers.Dense(128, activation='relu')

        # hidden layers
        self.hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.hidden_layer2 = tf.keras.layers.Dense(32, activation='relu')

        # output layer
        self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)

    # Called with either one element to determine next action
    def call(self, inputs, training=False, mask=None):
        x = self.input_layer(inputs)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        # x = self.hidden_layer3(x)
        return self.output_layer(x)