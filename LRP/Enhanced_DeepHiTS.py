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
sys.path.append("modules")

#basic libraries
import tensorflow as tf
import numpy as np

#model layers import
import sequential as sequential
import linear as linear
import convolution as convolution
import maxpool as maxpool
import avgCyclicPool2 as avgCyclicPool2
from rotAugment import augment_with_rotations

class DeepHiTS(object):

    """
    Cosntructor
    """
    def __init__(self, input_channels = 3):
        #tf.reset_default_graph()

        #number of input channels of a sample
        self.input_channels = input_channels
        
        #trainning step counter
        self.train_step = 0
        
        #initializate model graph
        self.input_batch, self.input_label, self.logits, self.output_probabilities, self.output_pred_cls = self._model_init()
        #initialize loss
        self.loss = self._loss_init()
        self.train_step = self._optimizer_init()

        self.sess = self._session_init()
        self.saver = tf.train.Saver()
        self._variables_init()
        
    #TODO:
    #create a trainning script
        
    """
    Gives predicted class for X that must have dimensions [None, 21, 21, 3]
    """
    def predict(self,X):
        try:
            return self.sess.run(self.output_pred_cls,
                                 feed_dict={
                                     self.dropout_prob: 1.0,
                                     self.input_batch: X
                                 })
            
        except ValueError:
            print(ValueError)
            
    """
    Gives probability estimates for X (softmax output) that have dimensions [None, 21, 21, 3]
    """
    def predict_proba(self,X):
        try:
            return self.sess.run(self.output_probabilities,
                                 feed_dict={
                                     self.dropout_prob: 1.0,
                                     self.input_batch: X
                                 })
            
        except ValueError:
            print(ValueError)
            
    """
    Save params to path, you can choose between saving the tensorflow graph or the params as numpy files
    """
    def get_params(self, path, as_graph=False, as_numpy=True):
        if(as_numpy and not(as_graph)):
            self._save_numpy_weights(path)
        else:
            self._save_checkpoint(path)
            
    """
    Load params from path, you can choose between load tensorflow graph or params as numpy files
    """
    def set_params(self, path, as_graph=False, as_numpy=True):
        if(as_numpy and not(as_graph)):
            self._load_numpy_weights(path)
        else:
            self._load_checkpoint(path)
        
    """
    initialization of model's variables
    """    
    def _variables_init(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init_new_vars_op = tf.variables_initializer([self.learning_rate])
        self.sess.run([init_op, init_new_vars_op])
    
    """
    Session initializer to allow more that one active session and GPU usage if available
    TODO: Check if there can be multiple models in one script 
    """    
    def _session_init(self):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    
    def fit(self, X, y):
        self.sess.run(self.train_step, 
                      feed_dict={self.input_batch: X, 
                                 self.input_label: y,
                                 self.dropout_prob: 0.5})
        
    def _load_checkpoint(self, checkpoint_path):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_path))

    def _save_checkpoint(self, checkpoint_path):
        self.saver.save(self.sess, checkpoint_path)
      
    #TODO gt weights to avoid except pass and lose knowing error sources
    def _load_numpy_weights(self, params_path):
        #values passed by reference
        W,B = self.net.getWeights()
        weights, biases = self.sess.run([W,B])
        counter_CNN = 1
        counter_FC = 1
        for i in range(len(W)):
            if(len(W[i].get_shape().as_list())==4):
                modify_op1 = W[i].assign(np.load(params_path+'/CNN'+str(counter_CNN)+'-W.npy'))
                modify_op2 = B[i].assign(np.load(params_path+'/CNN'+str(counter_CNN)+'-B.npy'))
                self.sess.run((modify_op1,modify_op2))
                counter_CNN +=1
            if(len(W[i].get_shape().as_list())==2):
                modify_op1 = W[i].assign(np.load(params_path+'/FC'+str(counter_FC)+'-W.npy'))
                modify_op2 = B[i].assign(np.load(params_path+'/FC'+str(counter_FC)+'-B.npy'))
                self.sess.run((modify_op1,modify_op2))
                counter_FC +=1


    def _save_numpy_weights(self, params_path):
        W,B = self.net.getWeights()
        weights, biases = self.sess.run([W,B])
        counter_CNN = 1
        counter_FC =1
        for i in range(len(W)):
            if(len(W[i].get_shape().as_list())==4):
                    np.save(params_path+'/CNN'+str(counter_CNN)+'-W.npy', np.array(weights[i]))
                    np.save(params_path+'/CNN'+str(counter_CNN)+'-B.npy', np.array(biases[i]))
                    counter_CNN +=1
            if(len(W[i].get_shape().as_list())==2):
                    np.save(params_path+'/FC'+str(counter_FC)+'-W.npy', np.array(weights[i]))
                    np.save(params_path+'/FC'+str(counter_FC)+'-B.npy', np.array(biases[i]))
                    counter_FC +=1

    """
    Init model from inputs to outputs
    """
    def _model_init(self):
        with tf.device('/cpu:0'):
            # make placeholder for feeding in data labels during training and evaluation
            input_label = tf.placeholder(shape=None, dtype=tf.int64, name="label")
            #one_hot_labels = tf.one_hot(labels, 2, dtype=tf.float32)
            # make placeholder for feeding in data during training and evaluation
            input_batch = tf.placeholder(shape=[None, 21, 21, self.input_channels], dtype=tf.float32, name="input")
        #generate rotated images
        augmented_input = augment_with_rotations(input_batch)
        #zero-pad images from 21x21 stamps to 27x27, to fit a 4x4 filter
        padded_input = tf.pad(augmented_input,
                              paddings=[
                                  [0, 0],
                                  [3, 3],
                                  [3, 3],
                                  [0, 0]])
    
        #Dropout placeholder
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout-prob')
        self.net = self._layers_init()       
        #feed-forward input and get logits output.
        logits = self.net.forward(padded_input)
        #pass logits through softmax for network output ([0,1] probability)
        #TODO: add softmax to modules
        output_probabilities = tf.nn.softmax(logits)
        #predicted classes
        output_pred_cls = tf.argmax(output_probabilities, 1)
        #true classes
        #y_true_cls = tf.argmax(y_, 1)

        return input_batch, input_label, logits, output_probabilities, output_pred_cls

    """
    define layers of model
    """
    def _layers_init(self):
        return sequential.Sequential([convolution.Convolution(kernel_size=4,
                                                                  output_depth=32, 
                                                                  input_depth=4,
                                                                  input_dim=27, act ='relu',
                                                                  stride_size=1, pad='VALID'),
                                    
                                        
                           convolution.Convolution(kernel_size=3, output_depth=32,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                           
                           maxpool.MaxPool(),
    
                           convolution.Convolution(kernel_size=3, output_depth=64,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                                        
                           convolution.Convolution(kernel_size=3, output_depth=64,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                                        
                           convolution.Convolution(kernel_size=3, output_depth=64,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                           maxpool.MaxPool(),
                           
                           linear.Linear(64, act ='relu', keep_prob=self.dropout_prob, 
                                           use_dropout = True),
                                                   
                           linear.Linear(64, act ='relu', keep_prob=self.dropout_prob, 
                                           use_dropout = True),
                                        
                           avgCyclicPool2.CyclicAvgPool(),
    
                           linear.Linear(2, act ='linear')])




    def _loss_init(self):
        labels = self.input_label
        self.one_hot_labels = tf.one_hot(labels, 2, dtype=tf.float32)
        self.diff = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_labels)
        loss = tf.reduce_mean(self.diff)
        
        return loss


    def _optimizer_init(self, learning_rate=0.04):
        #learning rate that can be actualized through trainning
        self.learning_rate = tf.Variable(learning_rate, trainable=False, collections=[])
        #SDG optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #train operation to be run for performing a leraning iteration
        train_step = optimizer.minimize(self.loss)
        return train_step
    

if __name__ == "__main__":
    sys.path.append("/home/ereyes/Alerce/AlerceDHtest/modules")
    from hits2013 import DH_set
    
    BATCH_SIZE = 1000
    path_dh =  '/home/ereyes/LRPpaper/datasets'
    path_weights = '/home/ereyes/LRPpaper/weights/CAP3c'
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