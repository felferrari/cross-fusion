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

#load the params-models.json options
with open(os.path.join('v1', 'params-model.json')) as param_file:
    params_model = json.load(param_file)
    
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




#tiles generation

##Create folder
tiles_patch = params_patches['tiles_patch']
shutil.rmtree(tiles_patch, ignore_errors=True)

labels = np.expand_dims(np.load(os.path.join(label_raw, 'labels.npy')), axis=-1)

#create preview


v_split = np.array_split(labels, params_patches['tiles_h'], axis=1)
tiles = []
for part in v_split:
    h_split = np.array_split(part, params_patches['tiles_v'], axis=0)
    for tile in h_split:
        tiles.append(tile)

for idx, tile in enumerate(tiles):
    os.makedirs(os.path.join(tiles_patch, f'{idx+1:02d}'))
    np.save(os.path.join(tiles_patch, f'{idx+1:02d}', f'{idx+1:02d}.npy'), tile)



del labels
