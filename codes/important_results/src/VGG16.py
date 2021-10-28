#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import numpy as np
import tensorflow as tf
import scipy
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt
from scipy import stats

from tqdm.auto import tqdm


# ## Extracting features from VGG16

# In[ ]:

def run():
    return_results = {}
    img_size = 224
    vgg16 = VGG16(weights='imagenet', include_top=True, pooling='max', input_shape = (img_size, img_size, 3))


    # In[ ]:


    op = vgg16.get_layer('fc2').output
    ip = vgg16.input

    basemodel = keras.Model(
        inputs=ip,
        outputs=op,
    )

    basemodel.trainable=False


    # In[ ]:


    images = np.empty((50, img_size, img_size, 3))
    for i in range(50):
      image = load_img("data/" + str(i+1) +".jpg", target_size=(img_size, img_size))
      image = img_to_array(image)

      for j in range(img_size):
        for k in range(img_size):
          g = (image[j,k,0] + image[j,k,1] + image[j,k,2])/3
          image[j,k,0] = g
          image[j,k,1] = g
          image[j,k,2] = g

      images[i,:,:,:] = image

    images = preprocess_input(images)
    features = basemodel.predict(images, verbose=1)

    colors = np.empty((5, img_size, img_size, 3))
    for i in range(5):
      image = load_img("data/C" + str(i) + ".png", target_size=(img_size, img_size))
      image = img_to_array(image)
      colors[i,:,:,:] = image

    colors = preprocess_input(colors)
    colorfeatures = basemodel.predict(colors, verbose=1)

    features = features.reshape(50,-1)
    colorfeatures = colorfeatures.reshape(5,-1)

    data = np.genfromtxt('data/bandw.csv', skip_header = 6, delimiter=',')
    humanpred = np.zeros((50,5))
    for i in range(50):
      for j in range(56):
        color = int(data[j, i]);
        humanpred[i, color] = humanpred[i, color]+1

    prob = humanpred/56


    # In[ ]:


    inputimages = np.zeros((250,features.shape[1]))
    inputcolors = np.zeros((250,features.shape[1]))
    outputprob = np.zeros((250))

    for i in range(50):
      for j in range(5):
        inputimages[5*i+j,:] = features[i,:]

    for i in range(5):
      for j in range(50):
        inputcolors[5*j+i] = colorfeatures[i, :]

    outputprob = prob.reshape(250)


    # ## Model

    # In[ ]:


    def get_model(op_features=75):
      imageinputs = keras.Input(shape=(features.shape[1]), name="imageinputs")
      colorinputs = keras.Input(shape=(features.shape[1]), name="colorinputs")
      dense = layers.Dense(op_features, name = "linearlayer", activation='relu')
      imageresults = dense(imageinputs)
      colorresults = dense(colorinputs)
      result = tf.keras.layers.Dot(axes=1, normalize=True, name="dot")([imageresults, colorresults])

      model = keras.Model(
          inputs=[imageinputs, colorinputs],
          outputs=[result],
      )
      model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(lr=0.001), metrics = ['mean_squared_error'])

      return model

    def get_corr(output, outputprob, vs, ve):
      x = np.copy(output.reshape(50,5))

      for i in range(50):
        rowsum = np.sum(x[i,:])
        for j in range(5):
          x[i,j] = x[i,j]/rowsum

      y = np.copy(outputprob.reshape(50,5))
      testx = x.reshape(-1)[vs:ve]
      testy = y.reshape(-1)[vs:ve]

      return scipy.stats.pearsonr(testx, testy)[0], scipy.stats.pearsonr(testx, testy)[1], testx


    # In[ ]:


    Rp = []
    pvalue = []

    for i in tqdm(range(50)):
      corr = [] #This contains correlation corresponding to specific validation data
      pred = np.zeros(250) #This will contain the final prediction for each of the loop. Total 5 subset of validation 50 each.

      # 5-fold cross validation
      for vs in [0, 50, 100, 150, 200]:
        ve = vs + 50
        model = get_model(op_features=75)
        history = model.fit(
            {"imageinputs": inputimages[[*range(vs)] + [*range(ve, 250)]], "colorinputs": inputcolors[[*range(vs)] + [*range(ve, 250)]]},
            {"dot": outputprob[[*range(vs)] + [*range(ve, 250)]]},
            epochs=30,
            batch_size=10,
            shuffle=True,
            verbose=0,
        )
        output = model.predict({"imageinputs": inputimages, "colorinputs": inputcolors})
        R, _, pred[vs:ve] = get_corr(output, outputprob, vs, ve)
        corr.append(R)

      temp = get_corr(pred, outputprob, 0, 250)
      Rp.append(temp[0])
      pvalue.append(temp[1])


    return_results['mean'] = np.mean(Rp)
    return_results['std'] = np.std(Rp)

    return return_results
