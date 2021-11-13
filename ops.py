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
from skimage.util.shape import view_as_windows
from sklearn.metrics import confusion_matrix
from multiprocessing.pool import Pool
from itertools import repeat


def generate_patches(img, size, stride):
    temp_image = np.pad(img, ((size, size), (size, size), (0, 0)), 'symmetric')
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
    temp_image = np.pad(img, ((size, size), (size, size), (0, 0)), 'symmetric')
    overlap = int((size-stride)/2)
    i = 0
    for line in tqdm(range(m.ceil(img.shape[0]/stride))):
        for col in range(m.ceil(img.shape[1]/stride)):
            i += 1
            l0 = size+line*stride-overlap
            c0 = size+col*stride-overlap
            patch = temp_image[l0:l0+size, c0:c0+size, :]
            np.save(os.path.join(save_path, f'{prefix}_{i:07d}'), patch)


def reconstruct_image(patches, stride, shape):
    n_lin = m.ceil(shape[0]/stride)
    n_col = m.ceil(shape[1]/stride)
    reconstructed_img = np.zeros(
        (n_lin*stride, n_col*stride, patches.shape[-1]))
    for line in range(n_lin):
        for col in range(n_col):
            reconstructed_img[line*stride:line*stride+stride, col*stride:col *
                              stride+stride] = crop_img(patches[col+line*n_col], stride)
    return reconstructed_img[:shape[0], :shape[1], :]


def reconstruct_image_from_path(path, stride, shape):
    patches_f = os.listdir(path)
    depth = np.load(os.path.join(path, patches_f[0])).shape[-1]

    n_col = m.ceil(shape[1]/stride)
    n_lin = m.ceil(shape[0]/stride)
    reconstructed_img = np.zeros((n_lin*stride, n_col*stride, depth))
    for line in range(n_lin):
        for col in range(n_col):
            patch = np.load(os.path.join(path, patches_f[col+line*n_col]))
            reconstructed_img[line*stride:line*stride+stride, col *
                              stride:col*stride+stride] = crop_img(patch, stride)
    return reconstructed_img[:shape[0], :shape[1], :]


def crop_img(img, final_size):
    crop_size = int((img.shape[0] - final_size)/2)
    return img[crop_size:crop_size+final_size, crop_size:crop_size+final_size, :]

def create_idx_image(ref_mask):
    return  np.arange(ref_mask.shape[0] * ref_mask.shape[1]).reshape(ref_mask.shape[0] , ref_mask.shape[1])

def extract_patches(im_idx, patch_size, overlap):
    '''overlap range: 0 - 1 '''
    row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
    return view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps))

def create_mask(size_rows, size_cols, grid_size=(6,3)):
    rows = np.array_split(np.arange(size_rows), grid_size[0])
    cols = np.array_split(np.arange(size_cols), grid_size[1])

    #num_tiles_rows = size_rows//grid_size[0]
    #num_tiles_cols = size_cols//grid_size[1]
    #print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    #patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((size_rows, size_cols))
    count = 0
    for row in rows:
        for col in cols:
            patch = np.ones((row.size, col.size))
            count += 1
            mask[row[0]:row[-1]+1, col[0]:col[-1]+1] = patch*count
    #plt.imshow(mask)
    #print('Mask size: ', mask.shape)
    return mask


def retrieve_idx_percentage(reference, patches_idx_set, patch_size, pertentage = 5):
    #count = 0
    new_idx_patches = []
    reference_vec = reference.reshape(reference.shape[0]*reference.shape[1])
    for patchs_idx in patches_idx_set:
        patch_ref = reference_vec[patchs_idx]
        class1 = patch_ref[patch_ref==1]
        if len(class1) >= int((patch_size**2)*(pertentage/100)):
            #count = count + 1
            new_idx_patches.append(patchs_idx)
    return np.asarray(new_idx_patches)

'''
Load the Optical Imagery -img-. Usually GDAL opens the image in [layers, height and width] order and need to be changed 
to [height, width and layers] order.
'''

def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            count+=1
    return img_reconstructed

def load_opt(img):
    return np.moveaxis(gdal.Open(img).ReadAsArray(), 0, 2)


def load_sar(img):
    temp = np.expand_dims(gdal.Open(img).ReadAsArray(), axis=-1)
    temp = 10**(temp/10)
    temp[temp > 1] = 1
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


def plot_layer(layer, inputs, to_file='model.png'):
    x = [Input(shape=inp) for inp in inputs]
    model = Model(x, layer.call(x))
    plot_model(model, to_file=to_file, show_shapes=True)

def matrics_AA_recall(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    thresholds = thresholds_    
    metrics_all = []
    
    for thr in thresholds:
        print(thr)  

        img_reconstructed = np.zeros_like(prob_map).astype(np.int8)
        img_reconstructed[prob_map >= thr] = 1
    
        mask_areas_pred = np.ones_like(ref_reconstructed)
        area = area_opening(img_reconstructed, area_threshold = px_area, connectivity=1)
        area_no_consider = img_reconstructed-area
        mask_areas_pred[area_no_consider==1] = 0
        
        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        #ref_no_consid = np.zeros((ref_reconstructed.shape))
        mask_borders[ref_reconstructed==2] = 0
        #mask_borders[ref_reconstructed==-1] = 0
        
        mask_no_consider = mask_areas_pred * mask_borders 
        ref_consider = mask_no_consider * ref_reconstructed
        pred_consider = mask_no_consider*img_reconstructed
        
        ref_final = ref_consider[mask_amazon_ts_==1]
        pre_final = pred_consider[mask_amazon_ts_==1]
        
        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        #TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)
        aa = (TP+FP)/len(ref_final)
        mm = np.hstack((recall_, precision_, aa))
        metrics_all.append(mm)
    metrics_ = np.asarray(metrics_all)
    return metrics_

def metric_thresholds(thr, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    print(thr)
    img_reconstructed = np.zeros_like(prob_map).astype(np.int8)
    img_reconstructed[prob_map >= thr] = 1

    mask_areas_pred = np.ones_like(ref_reconstructed)
    area = area_opening(img_reconstructed, area_threshold = px_area, connectivity=1)
    area_no_consider = img_reconstructed-area
    mask_areas_pred[area_no_consider==1] = 0
    
    # Mask areas no considered reference
    mask_borders = np.ones_like(img_reconstructed)
    #ref_no_consid = np.zeros((ref_reconstructed.shape))
    mask_borders[ref_reconstructed==2] = 0
    #mask_borders[ref_reconstructed==-1] = 0
    
    mask_no_consider = mask_areas_pred * mask_borders 
    ref_consider = mask_no_consider * ref_reconstructed
    pred_consider = mask_no_consider*img_reconstructed
    
    ref_final = ref_consider[mask_amazon_ts_==1]
    pre_final = pred_consider[mask_amazon_ts_==1]
    
    # Metrics
    cm = confusion_matrix(ref_final, pre_final)
    #TN = cm[0,0]
    FN = cm[1,0]
    TP = cm[1,1]
    FP = cm[0,1]
    precision_ = TP/(TP+FP)
    recall_ = TP/(TP+FN)
    aa = (TP+FP)/len(ref_final)
    mm = np.hstack((recall_, precision_, aa))
    return mm

def metrics_AP(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area, processes = 1):
    if processes > 1:
        pool = Pool(processes=processes)
        metrics = pool.starmap(
            metric_thresholds, 
            zip(
                thresholds_, 
                repeat(prob_map),
                repeat(ref_reconstructed),
                repeat(mask_amazon_ts_),
                repeat(px_area),
                )
            )
        return metrics
    else:
        metrics = []
        for thr in thresholds_:
            metrics.append(metric_thresholds(thr, prob_map, ref_reconstructed, mask_amazon_ts_, px_area))
            
        return metrics
            
    
def complete_nan_values(metrics):
    vec_prec = metrics[:,1]
    for j in reversed(range(len(vec_prec))):
        if np.isnan(vec_prec[j]):
            vec_prec[j] = 2*vec_prec[j+1]-vec_prec[j+2]
            if vec_prec[j] >= 1:
                vec_prec[j] == 1
    metrics[:,1] = vec_prec
    return metrics 


def recall_precision(y_true, y_pred, min_area, ths, mask=None):
    p_r = []
    for th in tqdm(ths):
        opt_class = np.zeros_like(y_pred, dtype=np.int8)
        opt_class[y_pred >= th] = 1

        mask_areas_pred = np.ones_like(y_pred)
        area_op = area_opening(opt_class, min_area)
        area_no_consider = opt_class-area_op  # 1- o que foi apagado pelo area_opening
        # 0- o que foi apagado pelo area_opening
        mask_areas_pred[area_no_consider == 1] = 0

        mask_borders = np.ones_like(opt_class)
        mask_borders[y_true == 2] = 0  # 0- tudo que for label=2

        # 0-tudo que for label=2 ou removido do area oppening 1-para o resto
        mask_no_consider = mask_areas_pred * mask_borders

        ref_consider = mask_no_consider * y_true
        pred_consider = mask_no_consider * opt_class

        if mask is not None:
            ref_final = ref_consider[mask == 1]
            pre_final = pred_consider[mask == 1]

        tp = np.count_nonzero(ref_final * pre_final)
        fp = np.count_nonzero(pre_final - pre_final * ref_final)
        fn = np.count_nonzero(ref_final - pre_final * ref_final)

        if tp+fp == 0:
            precision = 0
        else:
            precision = tp/(tp+fp)
        if tp+fn == 0:
            recall = 0
        else:
            recall = tp/(tp+fn)

        p_r.append([recall, precision])

    return np.array(p_r)


def compute_mAP(X, Y):
    # X -> Recall
    # Y -> Precision
    dx = np.diff(X)
    X_ = np.array([])
    Y_ = np.array([])

    eps = 5e-3
    for i in range(len(dx)):
        if dx[i] > eps:
            x0 = X[i]
            x1 = X[i+1]
            y0 = Y[i]
            y1 = Y[i+1]
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            x = np.arange(x0, x1, eps)
            y = a * x + b
            X_ = np.concatenate((X_, x))
            Y_ = np.concatenate((Y_, y))
        else:
            X_ = np.concatenate((X_, X[i:i+1]))
            Y_ = np.concatenate((Y_, Y[i:i+1]))

    X_ = np.concatenate((X_, X[-1:]))
    Y_ = np.concatenate((Y_, Y[-1:]))

    new_dx = np.diff(X_)
    map_ = 100 * np.inner(Y_[:-1], new_dx)

    return map_
