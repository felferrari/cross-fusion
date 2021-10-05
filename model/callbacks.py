# -*- coding: utf-8 -*-

from tensorflow.keras.callbacks import Callback
import tensorflow as tf

class UpdateAccuracy(Callback):
   
    def on_epoch_end(self, batch, logs = None):
        acc_opt = logs['opt_accuracy']
        acc_sar = logs['sar_accuracy']
        acc_fusion = logs['fusion_accuracy']
        
        den = acc_opt + acc_sar + acc_fusion + 1e-5

        self.model.accuracy_weights = tf.constant([(acc_opt + 1e-5)/den, (acc_sar + 1e-5)/den, (acc_fusion + 1e-5)/den])

