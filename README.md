# Fusion of SAR and optical imagery for deforestation in tropical rainforest

## Tensorflow
All code was tested with Tensorflow 2.4.0.

## JSON Configuration

The Model and training parameters of each experiment parameters are described in the .json files.

## Data Preparation

The Jupyter Notebook 00-Data Prep.ipynb prepare the images, concatenates all bands, normalizing the data of the two datasets: with or without clouds.

## Training Models

The Jupyter Notebooks 01_01_BASE_train.ipynb trains the single input/output models (UNET and ResUNET) using just one database (cloud OR no cloud).

The Jupyter Notebooks 01_01_BASE_train_mix.ipynb trains the single input/output models (UNET and ResUNET) using just both datasets (cloud AND no cloud).

The Jupyter Notebooks 02_01_FUS_train.ipynb trains the multipurpose models (Concat or CrossFusion) using just one database (cloud OR no cloud).

The Jupyter Notebooks 02_01_FUS_train_mix.ipynb trains the multipurpose models (Concat or CrossFusion) using just both datasets (cloud AND no cloud).

## Predictions

The Jupyter Notebooks 01_02_BASE_pred.ipynb generate the predictions of single input/output models (UNET and ResUNET).

The Jupyter Notebooks 02_02_FUS_pred.ipynb generate the predictions of multipurpose models (Concat or CrossFusion).

## Model Evaluation

The Jupyter Notebooks 01_03_BASE_eval.ipynb evaluate the Average Precision of the mean prediction maps and generate the Precision-Recall Curve 
of the single input/output models (UNET and ResUNET).

The Jupyter Notebooks 02_03_FUS_eval.ipynb evaluate the Average Precision of the mean prediction maps and generate the Precision-Recall Curve 
of the multipurpose models (Concat or CrossFusion).

## Experiment Review

The Jupyter Notebooks 01_04_BASE_review.ipynb review the experiment results from single input/output models (UNET and ResUNET).

The Jupyter Notebooks 02_04_FUS_review.ipynb review the experiment results from multipurpose models (Concat or CrossFusion).
