import tensorflow as tf
from model.models import Model_1
from dataloader import DataLoader
import os
import json
import shutil
from ops import summary, plot_layer
from model.layers import Encoder

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

shapes = [(128, 128, 8), (128, 128,4)]
model.summary(shapes)
model.plot(shapes)

shape = (128, 128, 8)
encoder = Encoder(name='encoder')
encoder.summary(shape)
encoder.plot(shape)
