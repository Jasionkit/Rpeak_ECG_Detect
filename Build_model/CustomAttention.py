import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

class CustomAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        print('input_shape={}'.format(input_shape))
        self.kernel = self.add_weight(
            shape=(1, *input_shape[1:-1], 1),
            initializer=initializers.Zeros(),
            name="kernel",
            trainable=True,
        )
    
    def call(self, inputs):
        return tf.multiply(inputs, tf.math.sigmoid(self.kernel)*2)