from ops import reconstruct_image
from tqdm import tqdm
import json
import  os
import numpy as np
from dataloader import DataLoader
from PIL import Image


# load the params-patches.json options
with open(os.path.join('v1', 'params-patches.json')) as param_file:
    params_patches = json.load(param_file)

# load the params-patches.json options
with open(os.path.join('v1', 'params-training.json')) as param_file:
    params_training = json.load(param_file)

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

dl_train = DataLoader(
    batch_size = params_training['batch_size'],
    opt_path=train_opt_patches,
    sar_path=train_sar_patches,
    label_path=train_label_patches,
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3,
    shuffle=True)

dl_val = DataLoader(
    batch_size=params_training['batch_size'],
    opt_path=val_opt_patches,
    sar_path=val_sar_patches,
    label_path=val_label_patches,
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3)

dl_test = DataLoader(
    batch_size=params_training['batch_size'],
    opt_path=test_opt_patches,
    sar_path=test_sar_patches,
    label_path=test_label_patches,
    patch_size=128,
    opt_bands=8,
    sar_bands=4,
    num_classes=3)

inp = dl_test.get_inputs()
rec_opt = reconstruct_image(inp[0], params_patches['patch_stride'], shape= (3547, 9202))
rec_sar = reconstruct_image(inp[1], params_patches['patch_stride'], shape= (3547, 9202))

im = Image.fromarray((255*rec_opt[:, :, :4]).astype(np.uint8))
im.save('opt_1.tif')

im = Image.fromarray((255*rec_opt[:, :, 4:]).astype(np.uint8))
im.save('opt_2.tif')

im = Image.fromarray(rec_sar[:, :, 0])
im.save('sar_1_1.tif')

im = Image.fromarray(rec_sar[:, :, 1])
im.save('sar_1_2.tif')

im = Image.fromarray(rec_sar[:, :, 2])
im.save('sar_2_1.tif')

im = Image.fromarray(rec_sar[:, :, 3])
im.save('sar_2_2.tif')

lab = dl_test.get_labels()
label = reconstruct_image(lab, params_patches['patch_stride'], shape= (3547, 9202))

im = Image.fromarray((255*label).astype(np.uint8))
im.save('label.tif')

print()