import json
import os
import numpy as np
from utils.utils import load_opt, min_max_scaler
from tifffile import tifffile


#load the params-tiles.json options
with open('params-tiles.json') as param_file:
    params_tiles = json.load(param_file)

img_path = params_tiles['img_path']
label_raw = os.path.join(img_path, params_tiles['label_sub'])

label = np.expand_dims(np.load(os.path.join(label_raw, 'labels.npy')), axis=-1)
print(label.shape)

tifffile.imsave(os.path.join(label_raw, 'labels.tif'), label)