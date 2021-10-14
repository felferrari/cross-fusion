import numpy as np
import math as m
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tensorflow.keras.models import Model
from model.layers import DataAugmentation
from PIL import Image
from osgeo import gdal
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from skimage.morphology import area_opening



def generate_patches(img, size, stride):
  temp_image = np.pad(img, ((size,size), (size,size), (0,0)), 'symmetric')
  overlap = int((size-stride)/2)
  patches = []
  for line in range(m.ceil(img.shape[0]/stride)):
    for col in range(m.ceil(img.shape[1]/stride)):
      l0 = size+line*stride-overlap
      c0 = size+col*stride-overlap
      patch = temp_image[l0:l0+size, c0:c0+size, :]
      patches.append(patch)
  
  return np.array(patches)

def generate_save_patches(img, size, stride, save_path, prefix):
  temp_image = np.pad(img, ((size,size), (size,size), (0,0)), 'symmetric')
  overlap = int((size-stride)/2)
  i = 0
  for line in tqdm(range(m.ceil(img.shape[0]/stride))):
    for col in range(m.ceil(img.shape[1]/stride)):
      i+=1
      l0 = size+line*stride-overlap
      c0 = size+col*stride-overlap
      patch = temp_image[l0:l0+size, c0:c0+size, :]
      np.save(os.path.join(save_path, f'{prefix}_{i:07d}'), patch)

def reconstruct_image(patches, stride, shape):
  n_lin = m.ceil(shape[0]/stride)
  n_col = m.ceil(shape[1]/stride)
  reconstructed_img = np.zeros((n_lin*stride, n_col*stride, patches.shape[-1]))
  for line in range(n_lin):
    for col in range(n_col):
      reconstructed_img[line*stride:line*stride+stride, col*stride:col*stride+stride] = crop_img(patches[col+line*n_col],stride)
  return reconstructed_img[:shape[0],:shape[1],:]

def reconstruct_image_from_path(path, stride, shape):
  patches_f = os.listdir(path)
  depth = np.load(os.path.join(path, patches_f[0])).shape[-1]

  n_col = m.ceil(shape[1]/stride)
  n_lin = m.ceil(shape[0]/stride)
  reconstructed_img = np.zeros((n_lin*stride, n_col*stride, depth))
  for line in range(n_lin):
    for col in range(n_col):
      patch = np.load(os.path.join(path, patches_f[col+line*n_col]))
      reconstructed_img[line*stride:line*stride+stride, col*stride:col*stride+stride] = crop_img(patch, stride)
  return reconstructed_img[:shape[0],:shape[1],:]

def crop_img(img, final_size):
  crop_size = int((img.shape[0] - final_size)/2)
  return img[crop_size:crop_size+final_size, crop_size:crop_size+final_size, :]

'''
Load the Optical Imagery -img-. Usually GDAL opens the image in [layers, height and width] order and need to be changed 
to [height, width and layers] order.
'''
def load_opt(img):
  return np.moveaxis(gdal.Open(img).ReadAsArray(), 0, 2)

def load_sar(img):
  temp = np.expand_dims(gdal.Open(img).ReadAsArray(), axis=-1)
  temp = 10**(temp/10)
  temp[temp>1] = 1
  return temp

def min_max_scaler(img):
  scaler = MinMaxScaler()
  shape = img.shape
  return scaler.fit_transform(np.expand_dims(img.flatten(), axis=-1)).reshape(shape)

def augment_data(data):
    aug = DataAugmentation()
    return aug.call(data).numpy()
    
def summary(layer, inputs):
  x = [Input(shape=inp) for inp in inputs]
  model = Model(x, layer.call(x))
  return model.summary()

def plot_layer(layer, inputs, to_file = 'model.png'):
  x = [Input(shape=inp) for inp in inputs]
  model = Model(x, layer.call(x))
  plot_model(model, to_file = to_file, show_shapes=True)

def recall_precision(y_true, y_pred, min_area, ths, mask = None):
  p_r = []
  for th in ths:
    opt_class = np.zeros_like(y_pred, dtype=np.int8)
    opt_class[y_pred>=th] = 1

    mask_areas_pred = np.ones_like(y_pred)
    area_op = area_opening(opt_class, min_area)
    area_no_consider = opt_class-area_op # 1- o que foi apagado pelo area_opening
    mask_areas_pred[area_no_consider==1] = 0 # 0- o que foi apagado pelo area_opening

    mask_borders = np.ones_like(opt_class)
    mask_borders[y_true==2] = 0 # 0- tudo que for label=2

    mask_no_consider = mask_areas_pred * mask_borders #0-tudo que for label=2 ou removido do area oppening 1-para o resto

    ref_consider = mask_no_consider * y_true
    pred_consider = mask_no_consider * opt_class

    if mask is not None:
      ref_final = ref_consider[mask==1]
      pre_final = pred_consider[mask==1]
    

    tp = np.count_nonzero(ref_final * pre_final)
    fp = np.count_nonzero(pre_final - pre_final * ref_final)
    fn = np.count_nonzero(ref_final - pre_final * ref_final)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    p_r.append([recall, precision])

  return p_r
    