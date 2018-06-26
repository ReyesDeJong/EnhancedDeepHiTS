#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class of Enhanced DeepHiTS model with sklearn standarization

LRP framework
Load parameters from numpy files

@author Esteban Reyes
"""

#python 2 and 3 comptibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import path to layers modules
import sys

#basic libraries
import numpy as np




if __name__ == "__main__":
    sys.path.append("/home/ereyes/Alerce/AlerceDHtest/modules")
    from hits2013 import DH_set
    from Enhanced_DeepHiTS import DeepHiTS
    
    BATCH_SIZE = 1000
    path_dh =  '/home/ereyes/LRPpaper/datasets'
    path_weights = 'weights/CAP_3c'
    #path_weights = '/home/ereyes/Alerce/AlerceDHtest/weights/CAP_ZTF/'
    
    DH_data = DH_set(data_path= path_dh, BATCH_SIZE=BATCH_SIZE)
    DH = DeepHiTS()
    #%%
    DH_array, DH_labels, _  = DH_data.get_test_sample()
    prediction = DH.predict(DH_array)
    accuracy_random = np.equal(DH_labels,prediction).mean()
    
    #%%
    DH.set_params(path_weights)
    prediction = DH.predict(DH_array)
    accuracy_pretrained = np.equal(DH_labels,prediction).mean()
    #%%
    DH.fit(DH_array, DH_labels)
    prediction = DH.predict(DH_array)
    accuracy_postrain = np.equal(DH_labels,prediction).mean()