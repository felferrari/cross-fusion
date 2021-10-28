
import numpy as np
from ops import reconstruct_image
from sklearn.metrics import classification_report
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.utils import to_categorical
import os

def test_model(model, dataset):
    y_pred = model.predict(x = dataset, verbose=1)
    y_opt = y_pred[0]
    y_sar = y_pred[1]
    y_fusion = y_pred[2]
    y_combined = y_pred[3]
    y_true = dataset.get_true()

    class_report_opt = eval_patches(y_true, y_opt)
    class_report_sar = eval_patches(y_true, y_sar)
    class_report_fusion = eval_patches(y_true, y_fusion)
    class_report_combined = eval_patches(y_true, y_combined)
    
    

    return class_report_opt, class_report_sar, class_report_fusion, class_report_combined


def eval_patches(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=3)
    y_true = np.argmax(y_true, axis=3)
    return classification_report(y_true.flatten(), y_pred.flatten(), output_dict=True)



def pred_patches(model, dataset, paths_to_pred, num_classes):
    for batch in tqdm(range(len(dataset))):
        pred_batch = model.predict(dataset.__getitem__(batch)[0], verbose=0)
        fnames_batch = dataset.get_names(batch)
        
        opt_batch = pred_batch[0]
        sar_batch = pred_batch[1]
        fusion_batch = pred_batch[2]
        combined_batch = pred_batch[3]
        
        opt_path = paths_to_pred[0]
        sar_path = paths_to_pred[1]
        fusion_path = paths_to_pred[2]
        combined_path = paths_to_pred[3]
        
        for idx, opt_img in enumerate(opt_batch):
            img_array = 255*to_categorical(np.expand_dims(np.argmax(opt_img, axis=2), axis=-1), num_classes)
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(os.path.join(opt_path, f'{fnames_batch[idx]}.tif'))
            
        for idx, sar_img in enumerate(sar_batch):
            img_array = 255*to_categorical(np.expand_dims(np.argmax(sar_img, axis=2), axis=-1), num_classes)
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(os.path.join(sar_path, f'{fnames_batch[idx]}.tif'))
            
        for idx, fusion_img in enumerate(fusion_batch):
            img_array = 255*to_categorical(np.expand_dims(np.argmax(fusion_img, axis=2), axis=-1), num_classes)
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(os.path.join(fusion_path, f'{fnames_batch[idx]}.tif'))
            
        for idx, combined_img in enumerate(combined_batch):
            img_array = 255*to_categorical(np.expand_dims(np.argmax(combined_img, axis=2), axis=-1), num_classes)
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(os.path.join(combined_path, f'{fnames_batch[idx]}.tif'))
            
            
            
        
        
        
        
            
            