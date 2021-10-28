# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Model Training
# %% [markdown]
# ## Importing

# %%
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from model.models import Model_1
from testing import test_model, pred_patches
from dataloader import DataLoader
from model.losses import FocalLoss, WBCE
from model.callbacks import UpdateAccuracy
from ops import reconstruct_image
import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import average_precision_score

# %% [markdown]
# ## Parameters

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
patches_path = params_patches['patches_path']

train_path = os.path.join(patches_path, params_patches['train_sub'])
val_path = os.path.join(patches_path, params_patches['val_sub'])
test_path = os.path.join(patches_path, params_patches['test_sub'])

# %% [markdown]
# ## Setting Dataloaders

# %%
dl_train = DataLoader(
    batch_size = params_training['batch_size'],
    data_path=os.path.join(train_path, params_patches['data_sub']),
    label_path=os.path.join(train_path, params_patches['label_sub']),
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3,
    shuffle=True#, 
    #limit=params_training['patch_limit']
)

dl_val = DataLoader(
    batch_size=params_training['batch_size'],
    data_path=os.path.join(val_path, params_patches['data_sub']),
    label_path=os.path.join(val_path, params_patches['label_sub']),
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3#,
    #limit=params_training['patch_limit']
)

dl_test = DataLoader(
    batch_size=params_training['batch_size'],
    data_path=os.path.join(test_path, params_patches['data_sub']),
    label_path=os.path.join(test_path, params_patches['label_sub']),
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3)

# %% [markdown]
# ## Model definition

# %%
model = Model_1(name='modelo_1')

metrics = {
}

weights = [0.2, 0.8, 0.0]

learning_rate = InverseTimeDecay(
    initial_learning_rate=1e-4, 
    decay_steps=params_training['learning_reduction']*len(dl_train),
    decay_rate = 0.01,
    staircase=True
    )

optimizers = {
    'opt': tf.keras.optimizers.Adam(learning_rate = learning_rate),
    'sar': tf.keras.optimizers.Adam(learning_rate = learning_rate),
    'fusion': tf.keras.optimizers.Adam(learning_rate = learning_rate),
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


# %%
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_combined_f1score',
        patience = params_training['patience'],
        mode = 'max',
        restore_best_weights=True),
    UpdateAccuracy()
]


history = model.fit(
    x=dl_train,
    validation_data=dl_val,
    epochs=params_training['epochs_train'],
    callbacks=callbacks,
    verbose = 2
    )

# %% [markdown]
# ## Show training history

# %%
plt.figure(figsize=(15, 8))
x = np.arange(len(history.history['loss']))+1
plt.plot(x, history.history['loss'], 'r-',label='Total Loss')
plt.plot(x, history.history['opt_loss'], 'r:',label='OPT Loss')
plt.plot(x, history.history['sar_loss'], 'r--',label='SAR Loss')
plt.plot(x, history.history['fusion_loss'], 'r-.',label='FUSION Loss')

plt.plot(x, history.history['val_loss'], 'g-',label='Total Val Loss')
plt.plot(x, history.history['val_opt_loss'], 'g:',label='OPT Val Loss')
plt.plot(x, history.history['val_sar_loss'], 'g--',label='SAR Val Loss')
plt.plot(x, history.history['val_fusion_loss'], 'g-.',label='FUSION Val Loss')

plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig('graphics/Loss.png')
plt.show()


# %%
plt.figure(figsize=(15, 8))
x = np.arange(len(history.history['loss']))+1
plt.plot(x, history.history['combined_accuracy'], 'r-',label='Combined Accuracy')
plt.plot(x, history.history['opt_accuracy'], 'r:',label='OPT Accuracy')
plt.plot(x, history.history['sar_accuracy'], 'r--',label='SAR Accuracy')
plt.plot(x, history.history['fusion_accuracy'], 'r-.',label='FUSION Accuracy')

plt.plot(x, history.history['val_combined_accuracy'], 'g-',label='Combined Val Accuracy')
plt.plot(x, history.history['val_opt_accuracy'], 'g:',label='OPT Val Accuracy')
plt.plot(x, history.history['val_sar_accuracy'], 'g--',label='SAR Val Accuracy')
plt.plot(x, history.history['val_fusion_accuracy'], 'g-.',label='FUSION Val Accuracy')

plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.savefig('graphics/Accuracy.png')
plt.show()


# %%
plt.figure(figsize=(15, 8))
x = np.arange(len(history.history['loss']))+1
plt.plot(x, history.history['combined_f1score'], 'r-',label='Combined F1 Score')
plt.plot(x, history.history['opt_f1score'], 'r:',label='OPT F1 Score')
plt.plot(x, history.history['sar_f1score'], 'r--',label='SAR F1 Score')
plt.plot(x, history.history['fusion_f1score'], 'r-.',label='FUSION F1 Score')

plt.plot(x, history.history['val_combined_f1score'], 'g-',label='Combined Val F1 Score')
plt.plot(x, history.history['val_opt_f1score'], 'g:',label='OPT Val F1 Score')
plt.plot(x, history.history['val_sar_f1score'], 'g--',label='SAR Val F1 Score')
plt.plot(x, history.history['val_fusion_f1score'], 'g-.',label='FUSION Val F1 Score')

plt.title('Training F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.savefig('graphics/F1score.png')
plt.show()

# %% [markdown]
# ## Evaluation 

# %%
opt_avg_prec_list = []
sar_avg_prec_list = []
fusion_avg_prec_list = []
combined_avg_prec_list = []


pred_path = params_patches['pred_path']
shutil.rmtree(pred_path, ignore_errors=True)
os.makedirs(pred_path)


for tile_n in params_patches['test_tiles']:
    dl_test.set_tile(int(tile_n))

    shape_tile = shapes_json[str(tile_n)]

    y_true = np.load(os.path.join(params_patches['tiles_path'], params_patches['label_sub'], f'label_{tile_n:02d}.npy'))
    y_true = to_categorical(y_true, 3)[:, :, 1]

    predictions_opt = []
    predictions_sar = []
    predictions_fusion = []
    predictions_combined = []

    for batch in tqdm(range(len(dl_test))):
        pred = model.predict_on_batch(dl_test[batch][0])
        predictions_opt.append(pred[0])
        predictions_sar.append(pred[1])
        predictions_fusion.append(pred[2])
        predictions_combined.append(pred[3])  

    predictions_opt = np.concatenate(predictions_opt, axis=0)  
    predictions_sar = np.concatenate(predictions_sar, axis=0)  
    predictions_fusion = np.concatenate(predictions_fusion, axis=0)  
    predictions_combined = np.concatenate(predictions_combined, axis=0)  

    predictions_opt_rec = reconstruct_image(predictions_opt, params_patches['patch_stride'], shape_tile)
    predictions_sar_rec = reconstruct_image(predictions_sar, params_patches['patch_stride'], shape_tile)
    predictions_fusion_rec = reconstruct_image(predictions_fusion, params_patches['patch_stride'], shape_tile)
    predictions_combined_rec = reconstruct_image(predictions_combined, params_patches['patch_stride'], shape_tile)

    np.save(os.path.join(params_patches['pred_path'], f'pred_opt_{tile_n:02d}.npy'), predictions_opt_rec)
    np.save(os.path.join(params_patches['pred_path'], f'pred_sar_{tile_n:02d}.npy'), predictions_sar_rec)
    np.save(os.path.join(params_patches['pred_path'], f'pred_fusion_{tile_n:02d}.npy'), predictions_fusion_rec)
    np.save(os.path.join(params_patches['pred_path'], f'pred_combined_{tile_n:02d}.npy'), predictions_combined_rec)

    opt_avg_prec = average_precision_score(y_true.flatten(), predictions_opt_rec[:,:,1].flatten())
    sar_avg_prec = average_precision_score(y_true.flatten(), predictions_sar_rec[:,:,1].flatten())
    fusion_avg_prec = average_precision_score(y_true.flatten(), predictions_fusion_rec[:,:,1].flatten())
    combined_avg_prec = average_precision_score(y_true.flatten(), predictions_combined_rec[:,:,1].flatten())

    opt_avg_prec_list.append(opt_avg_prec)
    sar_avg_prec_list.append(sar_avg_prec)
    fusion_avg_prec_list.append(fusion_avg_prec)
    combined_avg_prec_list.append(combined_avg_prec)

    print(f'Precision Average of OPT prediction of tile {tile_n} is {opt_avg_prec:.4f}')
    print(f'Precision Average of SAR prediction of tile {tile_n} is {sar_avg_prec:.4f}')
    print(f'Precision Average of FUSION prediction of tile {tile_n} is {fusion_avg_prec:.4f}')
    print(f'Precision Average of COMBINED prediction of tile {tile_n} is {combined_avg_prec:.4f}')



# %%



