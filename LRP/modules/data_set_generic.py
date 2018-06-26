#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:05:40 2018

Dataset Object

CHECK MAX DISBALANCE OPN REPLICATION FOR MULTICLASS

@author: ereyes
"""

import numpy as np
import matplotlib.pylab as plt

class dataset(object):
    
    """
    Cosntructor
    """
    def __init__(self, data_array, data_labels, BATCH_SIZE):
        self.BATCH_COUNTER = 0
        self.BATCH_COUNTER_EVAL = 0
        self.BATCH_SIZE = BATCH_SIZE
        self.data_array = data_array
        self.data_label = data_labels
        self.shuffle_data()
        
    def _merge_with_dataset(self, array, labels):
        self.data_label = np.concatenate((self.data_label, np.full(array.shape[0], labels)))
        self.data_array = np.concatenate((self.data_array,array))
        self.shuffle_data()
        
   #def load_json(self, path, labels = 0):
   #    data = self.data_preprocessor.get_preprocessed_data(path)
   #    self._merge_with_dataset(data, labels)
        

    def get_batch_images(self):
        batch, _ = self.get_batch()
        
        return batch
    
    def get_batch(self):
        if(self.BATCH_COUNTER+self.BATCH_SIZE<self.data_array.shape[0]):
            batch_image = self.data_array[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            batch_label = self.data_label[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            self.BATCH_COUNTER += self.BATCH_SIZE
            #print(get_batch.BATCH_COUNTER)
        else:
            self.BATCH_COUNTER = 0
            self.shuffle_data()
            batch_image = self.data_array[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            batch_label = self.data_label[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            
            #batch_image, batch_label = self.get_batch()
        
        return batch_image, batch_label
    
    def get_batch_eval(self):
        if(self.BATCH_COUNTER_EVAL+self.BATCH_SIZE<self.data_array.shape[0]):
            batch_image = self.data_array[self.BATCH_COUNTER_EVAL:self.BATCH_COUNTER_EVAL+self.BATCH_SIZE,...]
            batch_label = self.data_label[self.BATCH_COUNTER_EVAL:self.BATCH_COUNTER_EVAL+self.BATCH_SIZE,...]
            self.BATCH_COUNTER_EVAL += self.BATCH_SIZE
            #print(get_batch.BATCH_COUNTER)
        else:
            left_samples = self.data_array.shape[0]-self.BATCH_COUNTER_EVAL
            batch_image = self.data_array[self.BATCH_COUNTER:self.BATCH_COUNTER+left_samples,...]
            batch_label = self.data_label[self.BATCH_COUNTER:self.BATCH_COUNTER+left_samples,...]
            self.BATCH_COUNTER_EVAL = 0
            #self.shuffle_data()
            #batch_image, batch_label = self.get_batch()
        
        return batch_image, batch_label
    
    def shuffle_data(self):
        idx = np.arange(self.data_array.shape[0])
        np.random.shuffle(idx)        
        self.data_array = self.data_array[idx,...]
        self.data_label = self.data_label[idx,...]
        
    #TODO: change both values for uique functions (AVOID CODE REPLICATION)
    #TODO: recursively? replicate_data should be?
    #TODO: min_lbl_count changes on very iteration, it should stay the same or shuffle
    #of replicate_data cannot be
    """
    def balance_data_by_replication(self):
        labels = np.unique(self.data_label)
        max_lbl = np.max(labels) 
        min_lbl = np.min(labels)
        max_lbl_count= np.where(self.data_label==max_lbl)[0].shape[0]
        min_lbl_count = np.where(self.data_label==min_lbl)[0].shape[0]
        
        if(max_lbl_count==min_lbl_count):
            return
        
        elif (max_lbl_count-min_lbl_count > min_lbl_count):
            self.replicate_data(min_lbl, min_lbl_count)
            self.balance_data_by_replication()
        elif (max_lbl_count-min_lbl_count < min_lbl_count):
            self.replicate_data(min_lbl, max_lbl_count-min_lbl_count)
            self.balance_data_by_replication()
        return
    """

    def balance_data_by_replication(self):
        max_disbalance = self.get_max_disbalance()
        max_lbl_count, min_lbl_count  = self.get_max_min_label_count()
        max_lbl, min_lbl = self.get_max_min_label()
        
        if (max_disbalance==0):
            return
        
        while(max_disbalance!=0):
            if (min_lbl_count>max_disbalance):
                self.replicate_data(min_lbl, max_disbalance)
                #max_disbalance = 0
            else:
                self.replicate_data(min_lbl, min_lbl_count)
                #max_disbalance -= min_lbl_count 
                
            max_disbalance = self.get_max_disbalance()#
            
        self.shuffle_data()
        self.balance_data_by_replication()
        return    

    """
    def get_max_disbalance(self):
        labels = np.unique(self.data_label)
        disbalances = []
        for i in range(labels.shape[0]):
            label_i_count = np.where(self.data_label==labels[i])[0].shape[0]
            for j in range(labels.shape[0]):
                label_j_count = np.where(self.data_label==labels[j])[0].shape[0]
                disbalances.append(np.abs(label_i_count-label_j_count))
        
        disbalances = np.array(disbalances)
        max_disbalance = np.max(disbalances)-np.min(disbalances)
        return max_disbalance
    """
    def get_max_disbalance(self):
        max_label_count, min_label_count = self.get_max_min_label_count()
        return max_label_count-min_label_count
    """
    def get_max_min_label_count(self):
        labels = np.unique(self.data_label)
        labels_count = []
        
        for j in range(labels.shape[0]):
            label_j_count = np.where(self.data_label==labels[j])[0].shape[0]
            labels_count.append(label_j_count)
        
        labels_count = np.array(labels_count)
        return np.max(labels_count), np.min(labels_count)
    """
    def get_max_min_label_count(self):
        max_label, min_label = self.get_max_min_label()
        
        max_label_count = np.where(self.data_label==max_label)[0].shape[0]
        min_label_count = np.where(self.data_label==min_label)[0].shape[0]
        
        return max_label_count, min_label_count
    
    
    def get_max_min_label(self):
        labels = np.unique(self.data_label)
        labels_count = []
        
        for j in range(labels.shape[0]):
            label_j_count = np.where(self.data_label==labels[j])[0].shape[0]
            labels_count.append(label_j_count)
        
        labels_count = np.array(labels_count)
        
        max_label = labels[np.where(labels_count==np.max(labels_count))[0][0]]
        min_label = labels[np.where(labels_count==np.min(labels_count))[0][0]]
        return max_label, min_label
        
    
    def replicate_data(self, label, samples_number):
        #print("%i samples replicated of class %i" %(samples_number,label))
        label_idx = np.where(self.data_label==label)[0]
        np.random.shuffle(label_idx)
        label_idx = label_idx[0:samples_number]
        replicated_data_array = self.data_array[label_idx,...]
        self._merge_with_dataset(replicated_data_array, label)
        
    def get_array_from_label(self, label):
        label_idx = np.where(self.data_label==label)[0]
        return self.data_array[label_idx]
    
    #def print_sample(self, img):
    #    fig = plt.figure()
    #    for k, imstr in enumerate(['Template', 'Science', 'Difference']):
    #        ax = fig.add_subplot(1,3,k+1)
    #        ax.axis('off')
    #        ax.set_title(imstr)
    #        ax.matshow(img[...,k])