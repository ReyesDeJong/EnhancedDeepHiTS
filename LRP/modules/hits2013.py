#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:03:07 2018

HiTS2013 Dataset Tfrecords

@author: asceta
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DH_set(object):
    
    """
    Cosntructor
    """
    def __init__(self, data_path, BATCH_SIZE, sample_channels=3):
        #tf.reset_default_graph()
        self.sample_channels = sample_channels
        self.BATCH_SIZE = BATCH_SIZE
        self.data_path = data_path
        
        self.sess = self.session_init()
        #self.variables_init()
        self.define_set_tensors()
        self.create_threads()
        
    def define_set_tensors(self):
        #train tensors variables, run them to get dataset sample
        self.train_images, self.train_labels, self.train_snr = self.get_train_tensors()
        #validation tensors variables
        self.validation_images, self.validation_labels, self.validation_snr = self.get_validation_tensors()
        #test tensors variables
        self.test_images, self.test_labels, self.test_snr = self.get_test_tensors()
    """
    initialization of model's variables
    """    
    def variables_init(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        
    def create_threads(self):
        # Create a coordinator and run all QueueRunner objects for parallel loading of tfrecords
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        
    #funtion to get certain dataset tensor
    def get_dataset_tensor(self, dataset):
        data_path = self.data_path+dataset
        feature = {'image_raw': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64),
                   'snr': tf.FixedLenFeature([], tf.string)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path])
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image_raw'], tf.float32)
    
        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)
        
        # Cast snr data into int32
        snr = tf.decode_raw(features['snr'], tf.float64)
        
        # Reshape image data into the original shape
        image = tf.reshape(image, [21, 21, 4])
        
        #3 channel slice
        image_slice = tf.slice(image, [0, 0, 0], [21, 21, self.sample_channels]) 
        
        snr = tf.reshape(snr, [1])
    
        # Any preprocessing here ...
    
        # Creates batches by randomly shuffling tensors
        images, labels, snrs= tf.train.batch([image_slice, label, snr],
                                        batch_size=self.BATCH_SIZE,
                                        capacity=100000,
                                        num_threads=1)
        return images, labels, snrs
    
    def get_train_tensors(self):
        return self.get_dataset_tensor('/snr_train.tfrecord')
    
    
    def get_validation_tensors(self):
        return self.get_dataset_tensor('/snr_validation.tfrecord')
    
    
    def get_test_tensors(self):
        return self.get_dataset_tensor('/snr_test.tfrecord')
    
    
    
    
    def get_test_array(self):
        #test_images, test_labels, test_snr = self.get_test_tensors()
        images_array = self.sess.run(self.test_images)
        return images_array
    
    def get_test_sample(self):
        #test_images, test_labels, test_snr = self.get_test_tensors()
        images_array, label_array, snr_array = self.sess.run((self.test_images, self.test_labels, self.test_snr))
        return images_array, label_array, snr_array
    
    """
    Session initializer to allow more that one active session and GPU usage if available
    TODO: Check if there can be multiple models in one script 
    """    
    def session_init(self):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)