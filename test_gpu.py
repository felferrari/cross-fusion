
import tensorflow as tf

tf.get_logger().setLevel('DEBUG')
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))