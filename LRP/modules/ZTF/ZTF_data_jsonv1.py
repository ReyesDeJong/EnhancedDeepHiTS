#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:17:54 2018

ZTF Dataset

@author: asceta
"""
import numpy as np
import matplotlib.pylab as plt
import sys,os
sys.path.append(os.getcwd())
sys.path.append("..")
from data_set_generic import dataset

class ZTF_set(dataset):
    
    """
    Cosntructor
    """
    def __init__(self, data_path, BATCH_SIZE, data_preprocessor_obj, labels = 0):
        self.data_preprocessor = data_preprocessor_obj
        self.data_array = self.data_preprocessor.get_preprocessed_data(data_path)
        self.data_label = np.full(self.data_array.shape[0], labels)
        super().__init__(data_array=self.data_array, data_labels=self.data_label, BATCH_SIZE=BATCH_SIZE)
        self.shuffle_data()
        
    def load_json(self, path, labels = 0):
        data = self.data_preprocessor.get_preprocessed_data(path)
        self._merge_with_dataset(data, labels)
    
    def print_sample(self, img):
        fig = plt.figure()
        for k, imstr in enumerate(['Template', 'Science', 'Difference']):
            ax = fig.add_subplot(1,3,k+1)
            ax.axis('off')
            ax.set_title(imstr)
            ax.matshow(img[...,k])
        
        
#%%     
if __name__ == "__main__":
    from ZTF_preprocessor import ZTF_data_preprocessor
    
    path_data = '/home/ereyes/Alerce/AlerceDHtest/datasets/ZTF'
    path_reals = path_data+'/broker_reals.json'
    path_bogus = path_data+'/broker_bogus.json'
    
    ZTF_preproc = ZTF_data_preprocessor()
    
    ZTF_data = ZTF_set(data_path=path_reals, BATCH_SIZE=50, data_preprocessor_obj=ZTF_preproc, labels=1)
    #%%
    ZTF_data.load_json(path_bogus, labels=0)
    #%%
    ZTF_data.balance_data_by_replication()
    
    print("\nZTF data: %d samples" %ZTF_data.data_label.shape[0])
    print("ZTF data class 1: %d samples" %np.where(ZTF_data.data_label==1)[0].shape[0])
    print("ZTF data class 0: %d samples" %np.where(ZTF_data.data_label==0)[0].shape[0])
    
    