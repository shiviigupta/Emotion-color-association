#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import scipy
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt
from scipy import stats

from src.VGG16_Raw import run as VGG16_Raw
from src.VGG16 import run as VGG16
from src.VGG16_With_Wrong_Class import run as VGG16_With_Wrong_Class

import time

start = time.time()
print("Estimating Raw Correlation Score on VGG16 >>>>>>")
results_raw = VGG16_Raw(['fc2'])

print("Estimating Transformed Correlation Score on VGG16 >>>>>>")
results_transformed = VGG16()

print("## Printing Correlation Scores on Wrong labels >>>")
VGG16_With_Wrong_Class()
print("------------------------------------------------")

print("## Printing Correlation Scores on Actual Labels >>>")
print("Raw Score for fc2:", results_raw['fc2'])
print("Transformed Score for fc2: mean - ", results_transformed['mean'], "- std -", results_transformed['std'])
print()

print("Total eval time:", int(time.time()-start), "secs")
