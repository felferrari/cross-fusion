import json
import os
import numpy as np
import logging
from tqdm import tqdm

#load the params-patches.json options
with open(os.path.join('v1', 'params-patches.json')) as param_file:
    params_patches = json.load(param_file)

img_patches = params_patches['patches_path']

train_opt_patches = os.path.join(img_patches, params_patches['train_sub'] , params_patches['opt_sub'])
train_sar_patches = os.path.join(img_patches, params_patches['train_sub'] , params_patches['sar_sub'])
train_label_patches = os.path.join(img_patches, params_patches['train_sub'] , params_patches['label_sub'])

val_opt_patches = os.path.join(img_patches, params_patches['val_sub'] , params_patches['opt_sub'])
val_sar_patches = os.path.join(img_patches, params_patches['val_sub'] , params_patches['sar_sub'])
val_label_patches = os.path.join(img_patches, params_patches['val_sub'] , params_patches['label_sub'])

test_opt_patches = os.path.join(img_patches, params_patches['test_sub'] , params_patches['opt_sub'])
test_sar_patches = os.path.join(img_patches, params_patches['test_sub'] , params_patches['sar_sub'])
test_label_patches = os.path.join(img_patches, params_patches['test_sub'] , params_patches['label_sub'])

img_path = params_patches['img_path']
label_raw = os.path.join(img_path, params_patches['label_sub'])


#Analyze full data
count = [0, 0, 0]
label_files = os.listdir(label_raw)
for lf in tqdm(label_files):
    test_label = np.load(os.path.join(label_raw, lf))
    count[0] += np.count_nonzero(test_label == 0)
    count[1] += np.count_nonzero(test_label == 1)
    count[2] += np.count_nonzero(test_label == 2)

sum_values = count[0] + count[1] + count[2]
print()
print('========Full data analysis======')
print(f'Number of values 0: {100*count[0]/sum_values:.2f}%')
print(f'Number of values 1: {100*count[1]/sum_values:.2f}%')
print(f'Number of values 2: {100*count[2]/sum_values:.2f}%')



#analyze patches
count = [0, 0, 0]
label_files = os.listdir(train_label_patches)
for lf in tqdm(label_files):
    test_label = np.load(os.path.join(train_label_patches, lf))
    count[0] += np.count_nonzero(test_label == 0)
    count[1] += np.count_nonzero(test_label == 1)
    count[2] += np.count_nonzero(test_label == 2)

sum_values = count[0] + count[1] + count[2]
print()
print('=============Patches analysis==========')
print(f'Number of values 0: {100*count[0]/sum_values:.2f}%')
print(f'Number of values 1: {100*count[1]/sum_values:.2f}%')
print(f'Number of values 2: {100*count[2]/sum_values:.2f}%')
                               