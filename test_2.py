import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

#  Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype="int32")

# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype="int32")

# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)



model = keras.models.Model([text_input_a, text_input_b], encoded_input_a+encoded_input_b)
tf.keras.utils.plot_model(model)
model.summary()