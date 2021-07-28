#!/usr/bin/env python
# coding: utf-8

# # Model_prediction is the grand function that predicts the waterfall

# In[1]:
import numpy as np
import os
from pyuvdata import UVData
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy


import .utilities 
#get_ipython().run_line_magic('run', 'Utilities.ipynb')

#modul imort should come before file import
#rename module to lower case
#only import thing you use
#put all the functions in the cnn_rfi
# In[2]:


# # The model_prediction function
# ## If the user has the path to the directory and have not pre-loaded the data, use the model_prediction function

# In[8]:


def model_prediction(batch_num, batch_size, file_path, model_path):
    '''
    The model_prediction function takes in the data and use the machine learning model to categorize
    the sky data and the noise.
    Example Input:
    batch_num = 4
    batch_size = 30
    file_path = '/lustre/aoc/projects/hera/H4C/2459139'
    model_path = "/lustre/aoc/projects/hera/lchen/flexible_model.hdf5"

    Output:
    predicted data indicating sky data and signal data
    '''
    wf = utilities.get_directorydata(batch_num, batch_size, file_path)
    wf_renorm  = utilities.grand_data_renorm(wf)
    wf_reshape = wf_renorm.reshape(1, utilities.nextmultiple_time(wf), utilities.nextmultiple_freq(wf), 1)

    output_model = tf.keras.models.load_model(model_path)
    decoded_data = output_model.predict(wf_reshape)
    decoded_data_reshape = np.argmax(decoded_data, axis = -1)

    predicted_data = decoded_data_reshape[:, :len(wf[:,0]),:len(wf[0,:])]
    return predicted_data


# # The model_prediction_pre function
# ## If the user has the path to the directory and have not pre-loaded the data, use the model_prediction function

# In[11]:


def model_prediction_pre(data, file_path, model_path):
    '''
    The model_prediction function takes in the data and use the machine learning model to categorize
    the sky data and the noise.
    Example Input:
    data: pre-loaded data
    file_path = '/lustre/aoc/projects/hera/H4C/2459139'
    model_path = "/lustre/aoc/projects/hera/lchen/flexible_model.hdf5"

    Output:
    predicted data indicating sky data and signal data
    '''
    wf = data
    wf_renorm  = utilities.grand_data_renorm(wf)
    wf_reshape = wf_renorm.reshape(1, utilities.nextmultiple_time(wf), utilities.nextmultiple_freq(wf), 1)

    output_model = tf.keras.models.load_model(model_path)
    decoded_data = output_model.predict(wf_reshape)
    decoded_data_reshape = np.argmax(decoded_data, axis = -1)

    predicted_data = decoded_data_reshape[:, :len(wf[:,0]),:len(wf[0,:])]
    return predicted_data
