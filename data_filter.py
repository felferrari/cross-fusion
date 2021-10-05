import json
import os
import numpy as np
import logging
from tqdm import tqdm
from ops import augment_data


logging.basicConfig(filename = 'data_filter.log', level = logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


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

logging.info('==============Sarting filtering==============')
#Remove all data without label values 0 or 1 in 10% of data
must_have = [0,1]

logging.info('Removing training patches with less than 10% data')
#filter train
removed = 0
total = params_patches['patch_size'] * params_patches['patch_size']
min_val = int(0.1*total)
label_files = os.listdir(train_label_patches)
for lf in tqdm(label_files):
    test_label = np.load(os.path.join(train_label_patches, lf))
    if np.count_nonzero(test_label == must_have[0]) + np.count_nonzero(test_label == must_have[1]) < min_val:
        removed += 1
        os.remove(os.path.join(train_label_patches, lf))
        os.remove(os.path.join(train_opt_patches, lf))
        os.remove(os.path.join(train_sar_patches, lf))
        
logging.info(f'The filter removed {removed} training patches')

logging.info('Augmenting training patches with greater than 5% data')
aug = 0
min_val = int(0.05*total)
label_files = os.listdir(train_label_patches)
for lf in tqdm(label_files):
    data_label = np.load(os.path.join(train_label_patches, lf))
    if np.count_nonzero(data_label == must_have[1]) > min_val:
        
        data_opt = np.load(os.path.join(train_opt_patches, lf))
        data_sar = np.load(os.path.join(train_sar_patches, lf))
        
        aug_data_label = augment_data(data_label)
        aug_data_opt = augment_data(data_opt)
        aug_data_sar = augment_data(data_sar)
        
        for aug_idx in range(aug_data_label.shape[0]):
            np.save(os.path.join(train_label_patches, f'{lf[:-4]}_{aug_idx}.npy'), aug_data_label[aug_idx])
            np.save(os.path.join(train_opt_patches, f'{lf[:-4]}_{aug_idx}.npy'), aug_data_opt[aug_idx])
            np.save(os.path.join(train_sar_patches, f'{lf[:-4]}_{aug_idx}.npy'), aug_data_sar[aug_idx])
               
        aug+=1
        
        
logging.info(f'The filter augmented {aug} patches')

logging.info('==============Patches filter ended==============')