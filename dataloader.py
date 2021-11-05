from tensorflow.keras.utils import Sequence
import os
import math as m
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class DataLoader(Sequence):
    def __init__(self, image, reference, size, n_classes):
        self.image = image