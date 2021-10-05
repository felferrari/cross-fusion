from tensorflow.keras.layers import Layer, Conv2D, Input
from tensorflow.keras import Model
import tensorflow as tf
import json
import os
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import regularizers

# load the params-model.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)

regularizer = tf.keras.regularizers.L2(0.0001)

class AtrousConv(Layer):
    def __init__(self, filters=256, **kwargs):
        super(AtrousConv, self).__init__(**kwargs)

        # Atrous Spatial Pyramid Pooling
        self.aspp1_0 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer = regularizer,
            name='aspp_conv_1_0'
        )
        self.aspp3_0 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            dilation_rate=3,
            padding='same',
            kernel_regularizer = regularizer,
            name='aspp_conv_3_0'
        )
        self.aspp3_1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            dilation_rate=6,
            padding='same',
            kernel_regularizer = regularizer,
            name='aspp_conv_3_1'
        )
        self.aspp3_2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            dilation_rate=9,
            padding='same',
            kernel_regularizer = regularizer,
            name='aspp_conv_3_2'
        )

        # Image Level pooling
        self.imglevel_conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer = regularizer,
            name='imglev_conv'
        )

        # merge pooling
        self.merge_batchnorm = tf.keras.layers.BatchNormalization(
            axis=-1,
            name='merge_bn'
        )

    def call(self, inputs, training=True):
        shape = tf.shape(inputs)

        resize_height = shape[1]
        resize_width = shape[2]

        # Spatial Pyramid
        f_aspp1_0 = self.aspp1_0(inputs, training=training)
        f_aspp3_0 = self.aspp3_0(inputs, training=training)
        f_aspp3_1 = self.aspp3_1(inputs, training=training)
        f_aspp3_2 = self.aspp3_2(inputs, training=training)

        # image level pooling
        f_img_avg = tf.math.reduce_mean(inputs, [1, 2], keepdims=True)
        f_img_avg = self.imglevel_conv2d(f_img_avg, training=training)
        f_img_avg = tf.image.resize(f_img_avg, [resize_height, resize_width])

        # merge
        out = tf.keras.layers.concatenate(
            [f_aspp1_0, f_aspp3_0, f_aspp3_1, f_aspp3_2, f_img_avg], axis=-1)
        out = self.merge_batchnorm(out, training=training)
        out = tf.keras.activations.relu(out)

        return out

class Decoder(Layer):

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.aspp_0 = AtrousConv(name='aspp_0')
        self.conv_0 = tf.keras.layers.Conv2D(
            filters=params_model['decoder']['filters'],
            kernel_size=1,
            padding='same',
            kernel_regularizer = regularizer,
            name='conv_0')
        self.bn_0 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_0')
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=params_model['decoder']['filters'],
            kernel_size=3,
            padding='same',
            kernel_regularizer = regularizer,
            name='conv_1')
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_1')
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=params_model['decoder']['filters'],
            kernel_size=3,
            padding='same',
            kernel_regularizer = regularizer,
            name='conv_2')
        self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_2')

    def call(self, inputs, skip_list, training=True):
        x = self.aspp_0(inputs, training=training)

        x = self.conv_0(x,  training=training)
        x = self.bn_0(x, training=training)
        x = tf.keras.activations.relu(x)

        size_skip = tf.shape(skip_list[0])[1]
        x = tf.image.resize(x, [size_skip, size_skip])
        x = tf.keras.layers.concatenate([skip_list[0], x], axis=-1)

        x = self.conv_1(x, training=training)
        x = self.bn_1(x, training=training)
        x = tf.keras.activations.relu(x)

        x = tf.image.resize(x, [params_model['patch_size'], params_model['patch_size']])

        x = self.conv_2(x, training=training)
        x = self.bn_2(x, training=training)
        x = tf.keras.activations.relu(x)
        
        return x

class Encoder(Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.conv_0 = tf.keras.layers.Conv2D(
            filters=params_model['encoder']['filter_0'],
            kernel_size=3,
            name='conv_0',
            kernel_regularizer = regularizer,
            padding='same'
        )
        self.res_layers = []  # list of tuples (layer, skip=True/False)
        for block, enc_params in enumerate(params_model['encoder']['res_blocks']):
            for res_layer in range(enc_params['size']):
                if res_layer == 0 and enc_params['downsize']:
                    layer = ResNetLayer(
                        filters=enc_params['filters'], downsample=True, name=f'res_layer_{block}_{res_layer}')
                else:
                    layer = ResNetLayer(
                        filters=enc_params['filters'], downsample=False, name=f'res_layer_{block}_{res_layer}')

                if res_layer == enc_params['size']-1:
                    self.res_layers.append((layer, enc_params['skip']))
                else:
                    self.res_layers.append((layer, 0))

    def call(self, inputs, training=True):
        x = self.conv_0(inputs, training=training)
        skip_list = []
        for res_layer in self.res_layers:
            x = res_layer[0](x, training=training)
            if res_layer[1]:
                skip_list.append(x)

        return x, skip_list

    def summary(self, inputs_shape):
        x = Input(shape=inputs_shape)
        model = Model(x, self.call(x))
        return model.summary()

    def plot(self, inputs_shape, to_file='encoder.png'):
        x = Input(shape=inputs_shape)
        model = Model(x, self.call(x))
        plot_model(model, to_file = to_file, show_shapes=True)

class ResNetLayer(Layer):
    def __init__(self, filters, downsample=False, **kwargs):
        super(ResNetLayer, self).__init__(**kwargs)
        self.downsample = downsample
        self.filters = filters
        self.bn_0 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_0')
        if self.downsample:
            self.conv_0_down = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_regularizer = regularizer,
                name='conv_0_down')
            self.conv_init_down = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                strides=2,
                padding='same',
                kernel_regularizer = regularizer,
                name='conv_0_down_init')
        else:
            self.conv_0_nodown = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_regularizer = regularizer,
                name='conv_0')
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_1')
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_regularizer = regularizer,
            name='conv_1')

    def call(self, inputs, training=True):
        x_inputs = inputs
        x = self.bn_0(x_inputs, training=training)
        x = tf.keras.activations.relu(x)
        if self.downsample:
            x = self.conv_0_down(x, training=training)
            x_inputs = self.conv_init_down(x_inputs, training=training)
        else:
            x = self.conv_0_nodown(x, training=training)
        x = self.bn_1(x,  training=training)
        x = tf.keras.activations.relu(x)
        x = self.conv_1(x, training=training)

        return tf.keras.layers.add([x, x_inputs])

class Classifier(Layer):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(params_model['classifier']['dropout'], name='drop_0')
        self.conv_0 = tf.keras.layers.Conv2D(
                filters=params_model['classifier']['classes'],
                kernel_size=1,
                padding='same',
                kernel_regularizer = regularizer,
                name='conv_0')

    def call(self, input, training=True):
        x = self.dropout(input, training=training)
        x = self.conv_0(x, training=training)
        return tf.keras.activations.softmax(x)

class FusionLayer(Layer):
    def __init__(self, type, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)
        self.type = type

    def call(self, inputs):
        if self.type == 'sum':
            return tf.keras.layers.add(inputs)

class DataAugmentation(Layer):
    def __init__(self, **kwargs):
        super(DataAugmentation, self).__init__(**kwargs)
        
    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=0)
        x_rot_1 = tf.image.rot90(x, k = 1)
        x_rot_2 = tf.image.rot90(x, k = 2)
        x_rot_3 = tf.image.rot90(x, k = 3)
        x_flip_h = tf.image.flip_left_right(x)
        x_flip_v = tf.image.flip_up_down(x)
        
        return tf.keras.layers.concatenate([
            x,
            x_rot_1,
            x_rot_2,
            x_rot_3,
            x_flip_h,
            x_flip_v], axis=0)
    
class CombinationLayer(Layer):
    def __init__(self, **kwargs):
        super(CombinationLayer, self).__init__(**kwargs)
    def call(self, inputs, weights):
        return tf.math.multiply(inputs[0], weights[0]) + tf.math.multiply(inputs[1], weights[1]) + tf.math.multiply(inputs[2], weights[2])