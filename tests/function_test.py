#!/usr/bin/env python
# coding: utf-8

# In[7]:

import .utilities
import .model_prediction
import .vis_utils
#get_ipython().run_line_magic('run', 'Utilities.ipynb')
#get_ipython().run_line_magic('run', 'Model_prediction.ipynb')
#get_ipython().run_line_magic('run', 'Data_Visualization_Utilities.ipynb')


# In[8]:


wf = utilities.get_directorydata(4, 20, '/lustre/aoc/projects/hera/H4C/2459139')


# In[9]:


vis_utils.plot_original_data(wf)


# In[10]:


data = model_prediction.model_prediction_pre(wf, '/lustre/aoc/projects/hera/H4C/2459139', "/lustre/aoc/projects/hera/lchen/flexible_model.hdf5")


# In[11]:


vis_utils.plot_decoded_data(data)
