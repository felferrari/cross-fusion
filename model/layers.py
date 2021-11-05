from tensorflow.keras.layers import Layer, Conv2D, Input, MaxPool2D, UpSampling2D, concatenate
from tensorflow.keras import Model
import tensorflow as tf
import json
import os
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import regularizers
import numpy as np

# load the params-model.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)

regularizer = tf.keras.regularizers.L2(1e-2)

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
        #self.dropout = tf.keras.layers.Dropout(params_model['classifier']['dropout'], name='drop_0')
        self.conv_0 = tf.keras.layers.Conv2D(
                filters=params_model['classes'],
                kernel_size=1,
                padding='same',
                #kernel_regularizer = regularizer,
                name='conv_0')

    def call(self, input, training=True):
        #x = self.dropout(input, training=training)
        x=input
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

class RandomDataAugmentation(Layer):
    def __init__(self, **kwargs):
        super(RandomDataAugmentation, self).__init__(**kwargs)
        self.random = np.random.choice([True, False], 3)

    def call(self, x, y):
        x_0 = x[0]
        x_1 = x[1]
        y = y
        if self.random[0]:
            x_0 = tf.image.flip_left_right(x_0)
            x_1 = tf.image.flip_left_right(x_1)
            y = tf.image.flip_left_right(y)

        if self.random[1]:
            x_0 = tf.image.flip_up_down(x_0)
            x_1 = tf.image.flip_up_down(x_1)
            y = tf.image.flip_up_down(y)

        if self.random[2]:
            k = np.random.randint(1, 4)
            x_0 = tf.image.rot90(x_0, k=k)
            x_1 = tf.image.rot90(x_1, k=k)
            y = tf.image.rot90(y, k=k)
       
        return ((x_0, x_1), y)

class RandomDataAugmentation2(Layer):
    def __init__(self, **kwargs):
        super(RandomDataAugmentation2, self).__init__(**kwargs)
        self.random = np.random.choice([True, False], 3)

    def call(self, x, y):
        y = y
        if self.random[0]:
            x_0 = tf.image.flip_left_right(x)

            y = tf.image.flip_left_right(y)

        if self.random[1]:
            x_0 = tf.image.flip_up_down(x)
            y = tf.image.flip_up_down(y)

        if self.random[2]:
            k = np.random.randint(1, 4)
            x_0 = tf.image.rot90(x, k=k)
            y = tf.image.rot90(y, k=k)
       
        return (x, y)
    
class CombinationLayer(Layer):
    def __init__(self, **kwargs):
        super(CombinationLayer, self).__init__(**kwargs)

        self.opt_w =  self.add_weight(initializer='zeros', trainable=False, name='opt_w', dtype=tf.float32)
        self.sar_w =  self.add_weight(initializer='zeros', trainable=False, name='sar_w', dtype=tf.float32)
        self.fus_w =  self.add_weight(initializer='zeros', trainable=False, name='fus_w', dtype=tf.float32)

        self.opt_w.assign(tf.constant((1/3)))
        self.sar_w.assign(tf.constant((1/3)))
        self.fus_w.assign(tf.constant((1/3)))


    def updateWeights(self, accuracies):
        acc_opt = accuracies['opt_accuracy']
        acc_sar = accuracies['sar_accuracy']
        acc_fus = accuracies['fus_accuracy']

        den = acc_opt + acc_sar + acc_fus + 1e-5

        w_opt = (acc_opt + 1e-5)/den
        w_sar = (acc_sar + 1e-5)/den
        w_fus = (acc_fus + 1e-5)/den

        self.opt_w.assign(w_opt)
        self.sar_w.assign(w_sar)
        self.fus_w.assign(w_fus)



    def call(self, inputs):
        opt_in = inputs[0]*self.opt_w
        sar_in = inputs[1]*self.sar_w
        fus_in = inputs[2]*self.fus_w

        return opt_in + sar_in + fus_in


class UNET_Encoder(Layer):
    def __init__(self, filters, **kwargs):
        super(UNET_Encoder, self).__init__(**kwargs)
        self.filters = filters

        self.conv1 = Conv2D(filters[0], (3,3), activation='relu', padding='same', name = 'conv1')
        self.conv2 = Conv2D(filters[1], (3,3), activation='relu', padding='same', name = 'conv2')
        self.conv3 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv3')

        self.maxpool1 = MaxPool2D((2,2), name = 'maxPool1')
        self.maxpool2 = MaxPool2D((2,2), name = 'maxPool2')
        self.maxpool3 = MaxPool2D((2,2), name = 'maxPool3')

        self.conv4 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv4')
        self.conv5 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv5')
        self.conv6 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv6')

    def call(self, inputs, training):
        d1 = self.conv1(inputs, training = training) #128
        p1 = self.maxpool1(d1, training = training) #64

        d2 = self.conv2(p1, training = training) #64
        p2 = self.maxpool2(d2, training = training) #32

        d3 = self.conv3(p2, training = training) #32
        p3 = self.maxpool3(d3, training = training) #16

        b1 = self.conv4(p3, training = training) #16
        b2 = self.conv5(b1, training = training) #16
        b3 = self.conv6(b2, training = training) #16

        return b3, d3, d2, d1

class UNET_Decoder(Layer):
    def __init__(self, filters, n_classes, **kwargs):
        super(UNET_Decoder, self).__init__(**kwargs)
        self.filters = filters

        self.conv7 = Conv2D(filters[2], (3,3), activation='relu', padding='same', name = 'conv7')
        self.conv8 = Conv2D(filters[1], (3,3), activation='relu', padding='same', name = 'conv8')
        self.conv9 = Conv2D(filters[0], (3,3), activation='relu', padding='same', name = 'conv9')

        self.upsamp1 = UpSampling2D(size = (2,2), name = 'upSamp1')
        self.upsamp2 = UpSampling2D(size = (2,2), name = 'upSamp2')
        self.upsamp3 = UpSampling2D(size = (2,2), name = 'upSamp3')

        #self.last_conv = Conv2D(n_classes, (1,1), activation='softmax')


    def call(self, inputs, training):
        u3 = self.upsamp1(inputs[0], training = training) #32
        u3 = self.conv7(u3, training = training) #32
        m3 = concatenate([inputs[1], u3]) #32

        u2 = self.upsamp2(m3, training = training) #64
        u2 = self.conv8(u2, training = training) #64
        m2 = concatenate([inputs[2], u2]) #64

        u1 = self.upsamp3(m2, training = training) #128
        u1 = self.conv9(u1, training = training) #128
        m1 = concatenate([inputs[3], u1]) #128

        #out = self.last_conv(m1, training = training)

        return m1# out

class Conv2D_BN_RELU(Layer):
    def __init__(self, filters, padding = 'same', **kwargs):
        super(Conv2D_BN_RELU, self).__init__(**kwargs)
        self.conv = Conv2D(
            filters, 
            (3,3), 
            padding=padding, 
            kernel_regularizer=regularizer,
            name = 'conv')
        self.bn = tf.keras.layers.BatchNormalization(name='bn')

    def call(self, inputs, training):
        x = self.conv(inputs, training = training)
        x = self.bn(x, training=training)
        return tf.keras.activations.relu(x)


class CrossFusion(Layer):
    def __init__(self, filters, **kwargs):
        super(CrossFusion, self).__init__(**kwargs)
        self.h1 = Conv2D_BN_RELU(filters[0], name = 'h1')
        self.h2 = Conv2D_BN_RELU(filters[0], name = 'h2')
        self.j1 = Conv2D_BN_RELU(filters[1], name = 'j1')
        self.j2 = Conv2D_BN_RELU(filters[2], name = 'j2')

        self.j3 = tf.keras.layers.Conv2D(
                filters=params_model['classes'],
                kernel_size=1,
                padding='same',
                name='j3')

        self.recon_losses = []

    def call(self, inputs, training):
        x1_0 = inputs[0]
        x2_0 = inputs[1]

        x1 = self.h1(x1_0, training = training)
        x2 = self.h2(x2_0, training = training)
        x12 = self.h1(x2_0, training = training)
        x21 = self.h2(x1_0, training = training)

        j1 = concatenate((x1+x12, x2+x21))
        j2 = concatenate((x1, x21))
        j3 = concatenate((x12, x2))

        f1_0 = self.j1(j1, training = training)
        f2_0 = self.j1(j2, training = training)
        f3_0 = self.j1(j3, training = training)

        f1_1 = self.j2(f1_0, training = training)
        f2_1 = self.j2(f2_0, training = training)
        f3_1 = self.j2(f3_0, training = training)

        o1 = self.j3(f1_1, training = training)
        o2 = self.j3(f2_1, training = training)
        o3 = self.j3(f3_1, training = training)

        self.recon_losses = [
            tf.math.reduce_mean(tf.math.pow(o2-o1, 2)),
            tf.math.reduce_mean(tf.math.pow(o3-o1, 2))
            ]

        return tf.keras.activations.softmax(o1)


