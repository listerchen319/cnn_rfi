#!/usr/bin/env python
# coding: utf-8

# In[13]:

from utilities.py import *
from Model_prediction.py import *
#get_ipython().run_line_magic('run', 'Utilities.ipynb')
#get_ipython().run_line_magic('run', 'Model_prediction.ipynb')


# In[5]:


import numpy as np
import os
from pyuvdata import UVData
import tensorflow as tf
tf.config.list_physical_devices("GPU")


# In[6]:


import matplotlib.pyplot as plt
import scipy


# In[20]:


def plot_original_data(wf):
    '''
    This function takes in the origianl data and put it into a waterfall plot
    Input: the UNPREPROCESSED waterfall data
    Output: the waterfall plot
    '''
    wf = grand_data_renorm(wf)
    fig, ax = plt.subplots()
    im = plt.imshow(wf, aspect = 'auto')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Time (Seconds)")
    plt.title("Waterfall Plot, Fixed Baseline, Fixed Polarization")
    plt.colorbar(im)
    plt.show()


# In[21]:


def plot_decoded_data(decoded_data):
    '''
    The plot_decoded_data takes in the decoded_data and returns the waterfall plot
    Input: data predicted by the model
    Output: the waterfall plot of all the flags
    '''
    fig, ax = plt.subplots()
    im = plt.imshow((decoded_data[0,:,:]), aspect = 'auto')
    plt.colorbar(im)
    plt.title("Predicted data")
    plt.xlabel('frequency(Hz)')
    plt.ylabel('time (Seconds)')
    plt.show()
