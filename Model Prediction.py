# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Model evaluation
# %% [markdown]
# ## Importing

# %%
import json
from google_drive_downloader import GoogleDriveDownloader as gdd
import shutil, os
from ops import load_opt, load_sar, generate_save_patches, min_max_scaler
import numpy as np
import logging
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import skimage.morphology
from sklearn.metrics import confusion_matrix
from model.models import Model_1
from model.losses import FocalLoss, WBCE
from tensorflow.keras.layers import Input
from dataloader import DataLoader
from tensorflow.keras.optimizers.schedules import InverseTimeDecay

# %% [markdown]
# ## Load Model

# %%
# load the params-patches.json options
with open(os.path.join('v1', 'params-patches.json')) as param_file:
    params_patches = json.load(param_file)

# load the params-patches.json options
with open(os.path.join('v1', 'params-training.json')) as param_file:
    params_training = json.load(param_file)
    
#load the params-model.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)

#load the shapes.json options
with open('shapes.json') as param_file:
    shapes_json = json.load(param_file)


# %%
#patches_path = params_patches['patches_path']

#train_path = os.path.join(patches_path, params_patches['train_sub'])
#val_path = os.path.join(patches_path, params_patches['val_sub'])
#test_path = os.path.join(patches_path, params_patches['test_sub'])
#full_path = params_patches['full_path']

img_path = params_patches['img_path']
data_raw = os.path.join(img_path, params_patches['data_sub']) 
label_raw = os.path.join(img_path, params_patches['label_sub'])

pred_path = params_patches['pred_path']


# %%
model = Model_1(name='modelo_1')

metrics = {
}

weights = [0.2, 0.8, 0.0]


optimizers = {
    'opt': tf.keras.optimizers.Adam(learning_rate = 1e-4),
    'sar': tf.keras.optimizers.Adam(learning_rate = 1e-4),
    'fusion': tf.keras.optimizers.Adam(learning_rate = 1e-4),
}

class_indexes = [0, 1]

model.compile(
    optimizers = optimizers,
    loss_fn = WBCE,
    metrics_dict = metrics,
    class_weights = weights,
    class_indexes = class_indexes,
    run_eagerly=params_training['run_eagerly']
)

model.build(
    input_shape = [
        (None, params_patches['patch_size'], params_patches['patch_size'], params_model['opt_channels']),
        (None, params_patches['patch_size'], params_patches['patch_size'], params_model['sar_channels'])
    ]
    )
model.load_weights('weights.h5')

# %% [markdown]
# ## Model Predictions
# %% [markdown]
# ### Load images

# %%
opt = np.load(os.path.join(data_raw, 'opt.npy'))
sar = np.load(os.path.join(data_raw, 'sar.npy'))

# %% [markdown]
# ### Make Predictions

# %%
pred = model.predict_from_patches(
    (opt, sar), 
    params_patches['patch_size'], 
    params_patches['patch_stride'], 
    16)


# %%
labels = np.load(os.path.join(label_raw, 'labels.npy'))
labels = to_categorical(labels, 3)[:,:,1]


# %%
a = np.squeeze(np.uint8(pred[0]*255))
a.shape


# %%
fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (25,8))


img = Image.fromarray(np.uint8(labels*255))
ax[0].axis('off')
ax[0].set_title(f'Label')
ax[0].imshow(img, cmap = 'gray')

img = Image.fromarray(np.squeeze(np.uint8(pred[0]*255)))
ax[1].axis('off')
ax[1].set_title(f'OPT')
ax[1].imshow(img, cmap = 'gray')

img = Image.fromarray(np.squeeze(np.uint8(pred[1]*255)))
ax[2].axis('off')
ax[2].set_title(f'SAR')
ax[2].imshow(img, cmap = 'gray')

img = Image.fromarray(np.squeeze(np.uint8(pred[2]*255)))
ax[3].axis('off')
ax[3].set_title(f'FUSION')
ax[3].imshow(img, cmap = 'gray')

img = Image.fromarray(np.squeeze(np.uint8(pred[3]*255)))
ax[4].axis('off')
ax[4].set_title(f'COMBINED')
ax[4].imshow(img, cmap = 'gray')

fig.show()


# %%
np.save(os.path.join(pred_path, 'opt.npy'), np.squeeze(pred[0]))
np.save(os.path.join(pred_path, 'sar.npy'), np.squeeze(pred[1]))
np.save(os.path.join(pred_path, 'fusion.npy'), np.squeeze(pred[2]))
np.save(os.path.join(pred_path, 'combination.npy'), np.squeeze(pred[3]))


