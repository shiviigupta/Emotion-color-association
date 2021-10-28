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

from src.Color_Similarity import run as Color_Similarity
from src.Standard import run as Standard_CE

import time

start = time.time()
print("Training Color_Similarity model >>>>>>")
results_cs = Color_Similarity()

print("\nTraining Standard cross entropy model >>>>>>")
results_ce = Standard_CE()

raw_similarity = "Raw Similarity | " + "wrt Human Prediction - " + str(results_cs['raw_acc_human']) + " | wrt Actual Class - " + str(results_cs['raw_acc_actual'])
transformed_similarity = "Transformed Similarity | " + "wrt Human Prediction - " + str(results_cs['transformed_acc_human_mean']) + " | wrt Actual Class - " + str(results_cs['transformed_acc_actual_mean'])
human_ce = "Standard Classification (Trained on Human Prediction) | " + "wrt Human Prediction - " + str(results_ce['standard_human_acc_human_mean']) + " | wrt Actual Class - " + str(results_ce['standard_human_acc_actual_mean'])
actual_ce = "Standard Classification (Trained on Actual Class) | " + "wrt Human Prediction - " + str(results_ce['standard_actual_acc_human_mean']) + " | wrt Actual Class - " + str(results_ce['standard_actual_acc_actual_mean'])

print(raw_similarity)
print(transformed_similarity)
print(human_ce)
print(actual_ce)

print()
print("Total eval time:", int(time.time()-start), "secs")
