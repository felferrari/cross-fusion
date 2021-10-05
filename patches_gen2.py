import json
from google_drive_downloader import GoogleDriveDownloader as gdd
import shutil, os
from ops import load_opt, load_sar, generate_save_patches, min_max_scaler
import numpy as np
import logging
from PIL import Image
import tensorflow as tf


logging.basicConfig(filename = 'patches_gen.log', level = logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

logging.info('==============Starting patches generation==============')

#load the params-patches.json options
with open(os.path.join('v1', 'params-patches.json')) as param_file:
    params_patches = json.load(param_file)
    
#load the params-patches.json options
with open(os.path.join('v1', 'params-download.json')) as param_file:
    params_download = json.load(param_file)

img_path = params_patches['img_path']
opt_raw = os.path.join(img_path, params_patches['opt_sub'])
sar_raw = os.path.join(img_path, params_patches['sar_sub'])
label_raw = os.path.join(img_path, params_patches['label_sub'])

logging.info('Downloading files')
if params_patches['download']:
    shutil.rmtree(img_path, ignore_errors=True)
    os.makedirs(opt_raw)
    os.makedirs(sar_raw)
    os.makedirs(label_raw)

    #download and save OPT files
    for f in params_download['files_opt']:
        gdd.download_file_from_google_drive(file_id=f['id'],
                                        dest_path=os.path.join(opt_raw, f['name']))

    #download and save SAR files
    for f in params_download['files_sar']:
        gdd.download_file_from_google_drive(file_id=f['id'],
                                        dest_path=os.path.join(sar_raw, f['name']))

    #download and save LABELS files
    for f in params_download['files_labels']:
        gdd.download_file_from_google_drive(file_id=f['id'],
                                        dest_path=os.path.join(label_raw, f['name']))


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

logging.info('Creating patches folders')
shutil.rmtree(img_patches, ignore_errors=True)
os.makedirs(train_opt_patches)
os.makedirs(train_sar_patches)
os.makedirs(train_label_patches)

os.makedirs(val_opt_patches)
os.makedirs(val_sar_patches)
os.makedirs(val_label_patches)

os.makedirs(test_opt_patches)
os.makedirs(test_sar_patches)
os.makedirs(test_label_patches)

#process OPT files
logging.info('processing OPT files')
opt2018 = load_opt(os.path.join(opt_raw, '2018.tif'))
opt2019 = load_opt(os.path.join(opt_raw, '2019.tif'))
opt = np.concatenate((opt2018, opt2019), axis=-1)
opt = min_max_scaler(opt)


#number of lines
l_n = opt.shape[params_patches['split_dir']]
l_train = int(l_n*params_patches['train_split'])
l_val = int(l_n*params_patches['val_split'])


train_opt = opt[:l_train, :, :]
val_opt = opt[l_train:l_train+l_val, :, :]
test_opt = opt[l_train+l_val:, :, :]
logging.info(f'Training OPT shape: {train_opt.shape}')
logging.info(f'Validating OPT shape: {val_opt.shape}')
logging.info(f'Testing OPT shape: {test_opt.shape}')

logging.info('saving training OPT patches')
generate_save_patches(
    train_opt, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=train_opt_patches)

logging.info('saving validation OPT patches')
generate_save_patches(
    val_opt, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=val_opt_patches)

logging.info('saving test OPT patches')
generate_save_patches(
    test_opt, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=test_opt_patches)

del opt2018, opt2019

#process SAR files
logging.info('Processing SAR files')
sarvv2018 = load_sar(os.path.join(sar_raw, '2018_vv.tif'))
sarvh2018 = load_sar(os.path.join(sar_raw, '2018_vh.tif'))
sarvv2019 = load_sar(os.path.join(sar_raw, '2019_vv.tif'))
sarvh2019 = load_sar(os.path.join(sar_raw, '2019_vh.tif'))
sar = np.concatenate((sarvv2018, sarvh2018, sarvv2019, sarvh2019), axis=-1)
sar = min_max_scaler(sar)

train_sar = sar[:l_train, :, :]
val_sar = sar[l_train:l_train+l_val, :, :]
test_sar = sar[l_train+l_val:, :, :]
logging.info(f'Training SAR shape: {train_sar.shape}')
logging.info(f'Validating SAR shape: {val_sar.shape}')
logging.info(f'Testing SAR shape: {test_sar.shape}')

logging.info('saving training SAR patches')
generate_save_patches(
    train_sar, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=train_sar_patches)

logging.info('saving validation SAR patches')
generate_save_patches(
    val_sar, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=val_sar_patches)

logging.info('saving test SAR patches')
generate_save_patches(
    test_sar, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=test_sar_patches)

del sarvv2018, sarvh2018, sarvv2019, sarvh2019

#process LABEL files
logging.info('Processing LABEL patches')
label = np.expand_dims(np.load(os.path.join(label_raw, 'labels.npy')), axis=-1)

train_label = label[:l_train, :, :]
val_label = label[l_train:l_train+l_val, :, :]
test_label = label[l_train+l_val:, :, :]
logging.info(f'Training LABEL shape: {train_label.shape}')
logging.info(f'Validating LABEL shape: {val_label.shape}')
logging.info(f'Testing LABEL shape: {test_label.shape}')


img = Image.fromarray((255*tf.keras.utils.to_categorical(test_label, 3)).astype(np.uint8))
img.save('test_label.tif')



logging.info('saving training LABEL patches')
generate_save_patches(
    train_label, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=train_label_patches)

logging.info('saving validation LABEL patches')
generate_save_patches(
    val_label, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=val_label_patches)

logging.info('saving test LABEL patches')
generate_save_patches(
    test_label, 
    size=params_patches['patch_size'], 
    stride=params_patches['patch_stride'],
    save_path=test_label_patches)

logging.info('==============Patches generation ended==============')