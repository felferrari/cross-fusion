import tensorflow as tf
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from model.models import Model_1
from testing import test_model, pred_patches
from dataloader import DataLoader
from model.losses import FocalLoss
from model.callbacks import UpdateAccuracy
import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np


# load the params-patches.json options
with open(os.path.join('v1', 'params-patches.json')) as param_file:
    params_patches = json.load(param_file)

# load the params-patches.json options
with open(os.path.join('v1', 'params-training.json')) as param_file:
    params_training = json.load(param_file)
    
#load the params-model.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)

img_patches = params_patches['patches_path']

train_opt_patches = os.path.join(
    img_patches, params_patches['train_sub'], params_patches['opt_sub'])
train_sar_patches = os.path.join(
    img_patches, params_patches['train_sub'], params_patches['sar_sub'])
train_label_patches = os.path.join(
    img_patches, params_patches['train_sub'], params_patches['label_sub'])

val_opt_patches = os.path.join(
    img_patches, params_patches['val_sub'], params_patches['opt_sub'])
val_sar_patches = os.path.join(
    img_patches, params_patches['val_sub'], params_patches['sar_sub'])
val_label_patches = os.path.join(
    img_patches, params_patches['val_sub'], params_patches['label_sub'])

test_opt_patches = os.path.join(
    img_patches, params_patches['test_sub'], params_patches['opt_sub'])
test_sar_patches = os.path.join(
    img_patches, params_patches['test_sub'], params_patches['sar_sub'])
test_label_patches = os.path.join(
    img_patches, params_patches['test_sub'], params_patches['label_sub'])


test_pred_patches = os.path.join(
    img_patches, params_patches['test_sub'], params_patches['pred_sub'])
test_opt_pred_patches = os.path.join(
    test_pred_patches, params_patches['opt_sub'])
test_sar_pred_patches = os.path.join(
    test_pred_patches, params_patches['sar_sub'])
test_fusion_pred_patches = os.path.join(
    test_pred_patches, params_patches['fusion_sub'])
test_combined_pred_patches = os.path.join(
    test_pred_patches, params_patches['combined_sub'])

shutil.rmtree(test_pred_patches, ignore_errors=True)
os.makedirs(test_pred_patches)
os.makedirs(test_opt_pred_patches)
os.makedirs(test_sar_pred_patches)
os.makedirs(test_fusion_pred_patches)
os.makedirs(test_combined_pred_patches)

dl_train = DataLoader(
    batch_size = params_training['batch_size'],
    opt_path=train_opt_patches,
    sar_path=train_sar_patches,
    label_path=train_label_patches,
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3,
    shuffle=True, 
    limit=params_training['patch_limit'])

dl_val = DataLoader(
    batch_size=params_training['batch_size'],
    opt_path=val_opt_patches,
    sar_path=val_sar_patches,
    label_path=val_label_patches,
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3,
    limit=params_training['patch_limit'])

dl_test = DataLoader(
    batch_size=params_training['batch_size'],
    opt_path=test_opt_patches,
    sar_path=test_sar_patches,
    label_path=test_label_patches,
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3)



model = Model_1(name='modelo_1')
#model.summary([(128, 128, 8), (128, 128,4)])

#train(model, dl_train, dl_val, 10, [0,1])

metrics = {
}

learning_rate = InverseTimeDecay(
    initial_learning_rate=0.1, 
    decay_steps=params_training['learning_reduction']*len(dl_train),
    decay_rate = 0.1,
    staircase=True
    )

optimizers = {
    'opt': tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum=0.9),
    'sar': tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum=0.9),
    'fusion': tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum=0.9),
}

model.compile(
    optimizers = optimizers,
    class_indexes=[0,1],
    loss_fn = FocalLoss,
    metrics_dict = metrics,
    run_eagerly=params_training['run_eagerly']
)

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
'''
pred_patches(model,
             dl_test,
             (test_opt_pred_patches,
              test_sar_pred_patches,
              test_fusion_pred_patches,
              test_combined_pred_patches
                 ),
             params_model['classifier']['classes'])
'''

print()