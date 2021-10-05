from tensorflow.keras.utils import Sequence
import os
import math as m
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def remove_sub(text):
    return text.split('.')[0][6:]

class DataLoader(Sequence):
    def __init__(self, batch_size, data_path, label_path, patch_size,
                 opt_bands, sar_bands, num_classes, shuffle = False, limit = None):
        self.data_path = data_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.opt_bands = opt_bands
        self.sar_bands = sar_bands
        self.num_classes = num_classes
        self.limit = limit

        #generate a list of files in shuffle order
        self.list_files = list(map(remove_sub,os.listdir(self.label_path)))
        
        if self.shuffle:
            random.shuffle(self.list_files)

        if self.limit is not None:
            self.used_files = self.list_files[:self.limit]
        else:
            self.used_files = self.list_files
        
        


    def __len__(self):
        return m.ceil(len(self.used_files)/self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.list_files)

        if self.limit is not None:
            self.used_files = self.list_files[:self.limit]
        else:
            self.used_files = self.list_files


    def __getitem__(self, idx):
        ret_files = self.used_files[idx*self.batch_size:(idx+1)*self.batch_size]
        X_opt = np.empty((len(ret_files), self.patch_size, self.patch_size, self.opt_bands))
        X_sar = np.empty((len(ret_files), self.patch_size, self.patch_size, self.sar_bands))
        Y_label = np.empty((len(ret_files), self.patch_size, self.patch_size, self.num_classes))
        for i, f in enumerate(ret_files):
            X_opt[i, :, :, :] = np.load(os.path.join(self.data_path, f'opt_{f}.npy'))
            X_sar[i, :, :, :] = np.load(os.path.join(self.data_path, f'sar_{f}.npy'))
            Y = np.load(os.path.join(self.label_path, f'label_{f}.npy'))

            Y_label[i, :, :, :] = tf.keras.utils.to_categorical(Y, num_classes=self.num_classes)

        return (
            (tf.convert_to_tensor(X_opt, dtype=tf.float32), tf.convert_to_tensor(X_sar, dtype=tf.float32)),
            tf.convert_to_tensor(Y_label, dtype=tf.float32)
            )
    
    def get_names(self, idx):
        return self.used_files[idx*self.batch_size:(idx+1)*self.batch_size]

    def get_labels(self, to_categorical = True):
        Y_label = np.empty((len(self.used_files), self.patch_size, self.patch_size, self.num_classes))
        pbar = tqdm(self.used_files)
        for i, f in enumerate(pbar):
            Y = np.load(os.path.join(self.label_path, f'label_{f}.npy'))
            if to_categorical:
                Y_label[i, :, :, :] = tf.keras.utils.to_categorical(Y, num_classes=self.num_classes)
            else:
                Y_label[i, :, :, :] = Y
        return Y_label

    def get_inputs(self):
        X_opt = np.empty((len(self.used_files), self.patch_size, self.patch_size, self.opt_bands))
        X_sar = np.empty((len(self.used_files), self.patch_size, self.patch_size, self.sar_bands))
        pbar = tqdm(self.used_files)
        for i, f in enumerate(pbar):
            X_o = np.load(os.path.join(self.opt_path, f'{f}.npy'))
            X_s = np.load(os.path.join(self.sar_path, f'{f}.npy'))
            X_opt[i, :, :, :] = X_o
            X_sar[i, :, :, :] = X_s
        return (X_opt, X_sar)

