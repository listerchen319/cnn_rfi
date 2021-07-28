#!/usr/bin/env python
# coding: utf-8

# # These functions are the utilities for model testing

# ### Modual requirements
# numpy
# os
# pyuvdata
# tensorflow

# In[1]:


import numpy as np
import os
from pyuvdata import UVData
import tensorflow as tf


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import scipy


# ## Get Data Directory

# In[95]:


def get_directorydata(batch_num, batch_size, file_path):
    '''
    The get_directorydata function load data with corresponding batch size and returns time and frequency data array
    Input:
    batch_num: which group of files one would like to load
    batch_size: the size of the batch one wants to load. Notices that the size of the time
    array would be 2*batch_size
    filepath: where the data is stored
    output: the waterfall data with time * frequency array

    Example Input:
    batch_num = 2
    batch_size = 24
    file_path =  '/lustre/aoc/projects/hera/H4C/2459139'
    Output:
    waterfall data of (48*1536)

    WILL return an error if:
    batch_num is less than 1

    Reading the data usually takes some time.
    '''
    #batch_size default 32, but test when batch_size is not a multiiple of 16
    filelist = [d for d in sorted(os.listdir(file_path)) if d.endswith(".sum.uvh5")]
    #this will get the *2nd* batch of files!
    batch = batch_num - 1
    idx0 = batch * batch_size
    idx1 = idx0 + batch_size
    uvd = UVData()
    filename = filelist[idx0 : idx1]
    #print(filename)
    full_file_path = []
    for names in filename:
        path = full_path = os.path.join(file_path, names)
        full_file_path.append(path)
    uvd.read(full_file_path, antenna_nums=[0,1], axis="blt")
    wf = uvd.get_data(0,1,'nn')
    return wf


# # Pre-process the data: Utilities

# ## This is to find the remainder of the time dimension

# In[96]:


def find_remainder_time(wf):
    '''
    This function will find the remainder when the time dimension of the waterfall is
    divided by 16
    Input: waterfall data
    Output:
    - If the shape is already a multiple of 16:
        The function returns statement: "Time is a multiple of 16" and return 0
    - If the shape is NOT a multiple of 16:
        The function returns the remainder of time shape divided by 16
    reminder!! This function only works when the number of time channel
    is larger than 9
    '''
    length_time = len(wf[:,0])
    #print(length_time)
    divider = int(length_time/16)
    #print(divider)
    remainder = length_time - divider*16
    if remainder == 0:
        print("Time is a multiple of 16")
        return 0
    return remainder


# ## This is to find the remainder of the frequency dimension

# In[97]:


def find_remainder_frequency(wf):
    '''
    This function will find the remainder when the frequency dimension of the waterfall is
    divided by 16
    Input: waterfall data
    Output:
    - If the shape is already a multiple of 16:
        The function returns statement: "Frequency is a multiple of 16" and return 0
    - If the shape is NOT a multiple of 16:
        The function returns the remainder of frequency shape divided by 16
    reminder!! This function only works when the number of frequency channel
    is larger than 9
    '''
    length_time = len(wf[0,:])
    #print(length_time)
    divider = int(length_time/16)
    #print(divider)
    remainder = length_time - divider*16
    if remainder == 0:
        print("Frequency is a multiple of 16")
        return 0
    return remainder


# ## This function find the next 16 multiple for time

# In[108]:


def nextmultiple_time(wf):
    '''
    This function takes the waterfall and returns the next multiple of 16 in the time channel

    reminder!! This function only works when the number of time channel
    is larger than 9
    '''
    length_time = len(wf[:,0])
    divider = int(length_time/16)
    nextmultiple = 16*(divider+1)
    remainder = length_time - divider*16
    if remainder == 0:
        print("Frequency is already is a multiple of 16, don't need to find the next multiple of 16")
        return len(wf[0,:])
    else:
        return nextmultiple


# ##  This function find the next 16 multiple of frequency

# In[99]:


def nextmultiple_freq(wf):
    '''
    This function takes the waterfall and returns the next multiple of 16 in the frequency channel

    reminder!! This function only works when the number of frequeny channel
    is larger than 9
    '''
    length_freq = len(wf[0,:])
    divider = int(length_freq/16)
    nextmultiple = 16*(divider+1)
    remainder = length_freq - divider*16
    if remainder == 0:
        print("Frequency is already is a multiple of 16, don't need to find the next multiple of 16")
        return len(wf[0,:])
    else:
        return nextmultiple


# ## Flip the waterfall

# In[104]:


def renorm(arr):
    '''
    This function takes in the original data and
    take the absolute value and log10 of the data
    Input: original data array
    '''
    #first we do the normal: abs and log 10
    arr = np.abs(arr)
    arr = np.log10(arr)
    return arr


#Raise an error if either dimension is less than 9!
def time_shaping(arr):
    '''
    This function takes the original shape of the data and augment the time
    axis to a multiple of 16
    Input: original data array
    Output: the augmented array which the time axis of the array will be augmented
    to a multiple of 16

    If the length of the time axis is already a multiple of 16, then the function will
    return the original time axis data untouched
    '''
    if find_remainder_time(arr) == 0:
        return arr
    #drop the last column in wf:
    else:
        wf_dropped = np.delete(arr, -1, axis = 0)
        #print(np.shape(wf_dropped))
        wf_flipped = np.flip(arr, axis = 0)
        #print(np.shape(wf_flipped))
        wf_full = np.concatenate((wf_dropped,wf_flipped), axis = 0)
        #print(np.shape(wf_full))
        wf_filling = wf_full[:nextmultiple_time(arr),:]
        print("the new shape is " + str(np.shape(wf_filling)))
        return wf_filling

def freq_shaping(arr):
    '''
    This function takes the original shape of the data and augment the frequency
    axis to a multiple of 16
    Input: original data array
    Output: the augmented array which the frequency axis of the array will be augmented
    to a multiple of 16

    If the length of the frequency axis is already a multiple of 16, then the function will
    return the original frequeny axis data untouched
    '''
    if find_remainder_frequency(arr) == 0:
        return arr
    else:
        #drop the last column in wf:
        wf_dropped = np.delete(arr, -1, axis = 1)
        #print(np.shape(wf_dropped))
        wf_flipped = np.flip(arr, axis = 1)
        #print(np.shape(wf_flipped))
        wf_full = np.concatenate((wf_dropped,wf_flipped), axis = 1)
        #print(np.shape(wf_full))
        wf_filling = wf_full[:,:nextmultiple_freq(arr)]
        print("the new shape is " + str(np.shape(wf_filling)))
        return wf_filling


# ## Grand_data_renorm: Run this function to complete all the data pre-processing

# In[105]:


def grand_data_renorm(wf):
    '''
    the grand_data_renorm function takes in a waterfall goes through all the process of pre-procesisng
    the data before putting it into the machine learning model
    '''
    wf_abs_log10 = renorm(wf)
    wf_timeshaping = time_shaping(wf_abs_log10)
    wf_freqshaping = freq_shaping(wf_timeshaping)
    return wf_freqshaping


# In[106]:


def load_model(model_path):
    '''
    Input:
    filepath: has to be a string of where the model is
    Example input:

    filepath = "/lustre/aoc/projects/hera/lchen/flexible_model.hdf5"
    '''
    output_model_full = tf.keras.models.load_model(model_path)
    return output_model_full


# In[107]:


def print_summary(model):
    '''
    output the summary of the machine learning model
    '''
    return model.summary()
