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

#basic libraries
import tensorflow as tf
import numpy as np

#model layers import
import modules.sequential32 as sequential32
import modules.linear32 as linear32
import modules.convolution32 as convolution32
import modules.maxpool32 as maxpool32
import modules.avgCyclicPool2 as avgCyclicPool2
from modules.rotAugment import augment_with_rotations

class DeepHiTS(object):

    """
    Cosntructor
    """
    def __init__(self):
        #number of samples for every trainning iteration
        self.BATCH_SIZE = 50
        
        #trainning step counter
        self.train_step = 0


        #self.data_train, self.data_test = self._data_init()
        
        #initializate model graph
        self.input_batch, self.input_label, self.output_probabilities, self.output_pred_cls = self._model_init()
        
        #self.recon_loss, self.auto_encoder_loss, self.disc_loss = self._loss_init()
        #self.ae_train_step, self.disc_train_step = self._optimizer_init()

        self.sess = self._session_init()
        self.saver = tf.train.Saver()
        self._variables_init()
        
    #TODO:get_params() and set_params(): Load/save state functionality
    #fit(X, y): Fit the model to data matrix X and target(s) y

        
    """
    Gives predicted class for X that must have dimensions [None, 21, 21, 4]
    """
    def predict(self,X):
        try:
            return sess.run(self.output_pred_cls,
                                 feed_dict={
                                     self.dropout_prob: 1.0,
                                     self.input_batch: X
                                 })
            
        except ValueError:
            print("Invalid input size of %s, should be of dimensions [None, 21, 21, 4]" 
                   % (str(x.shape)))
            
    """
    Gives probability estimates for X (softmax output) that have dimensions [None, 21, 21, 4]
    """
    def predict_proba(self,X):
        try:
            return sess.run(self.output_probabilities,
                                 feed_dict={
                                     self.dropout_prob: 1.0,
                                     self.input_batch: X
                                 })
            
        except ValueError:
            print("Invalid input size of %s, should be of dimensions [None, 21, 21, 4]" 
                   % (str(x.shape)))
        
    """
    initialization of model's variables
    """    
    def _variables_init(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
    
    """
    Session initializer to allow more that one active session and GPU usage if available
    TODO: Check if there can be multiple models in one script 
    """    
    def _session_init(self):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    """
    def train(self):
        print("Beginning training")
        it=0
        max_it = 100000
        saving_every = 1000
        while it < max_it:
            it += 1
            self.sess.run(self.ae_train_step, {self.input_ph: self.sample_minibatch()})
            self.sess.run(self.disc_train_step, {self.input_ph: self.sample_minibatch()})

            if it % 500 == 0:
                batch = self.sample_minibatch()
                ae_train_loss = self.sess.run(self.auto_encoder_loss, {self.input_ph: batch})
                recon_train_loss = self.sess.run(self.recon_loss, {self.input_ph: batch})
                disc_train_loss = self.sess.run(self.disc_loss, {self.input_ph: batch})
                print("Iteration %i: \n    Autoencoder loss (train) %f\n    Reconstruction loss (train) %f\n    Discriminator loss (train) %f" % (it, ae_train_loss, recon_train_loss, disc_train_loss), flush=True)
                print("Iteration %i: \n    Autoencoder loss (train) %f\n    Reconstruction loss (train) %f\n    Discriminator loss (train) %f" % (it, ae_train_loss, recon_train_loss, disc_train_loss), flush=True, file=open('train.log','a'))

                ae_test_loss = self.sess.run(self.auto_encoder_loss, {self.input_ph: self.data_test[0:500]})
                recon_test_loss = self.sess.run(self.recon_loss, {self.input_ph: self.data_test[0:500]})
                disc_test_loss = self.sess.run(self.disc_loss, {self.input_ph: self.data_test[0:500]})
                print("    Autoencoder loss (test) %f\n    Reconstruction loss (test) %f\n    Discriminator loss (test) %f" % (ae_test_loss, recon_test_loss, disc_test_loss), flush=True)
                print("    Autoencoder loss (test) %f\n    Reconstruction loss (test) %f\n    Discriminator loss (test) %f" % (ae_test_loss, recon_test_loss, disc_test_loss), flush=True, file=open('train.log','a'))

            if it % saving_every == 0:
                model_path = "checkpoints/model"
                save_path = self.saver.save(self.sess, model_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open('train.log','a'))
        #model_path = "checkpoints/model"
        #save_path = self.saver.save(self.sess, model_path, global_step=it)
        #print("Model saved to: %s" % save_path)
        #print("Model saved to: %s" % save_path, file=open('train.log', 'a'))
        """
        
    def load_latest_checkpoint(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint('checkpoints'))

    """
    def sample_minibatch(self, batch_size=64, test=False):
        if test is False:
            indices = np.random.choice(range(len(self.data_train)), batch_size, replace=False)
            sample = self.data_train[indices]
        elif test is True:
            indices = np.random.choice(range(len(self.data_test)), batch_size, replace=False)
            sample = self.data_test[indices]
        return sample

    def make_plots(self):
        pass

    def _data_init(self):
        # dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz must be in the root
        # folder. Find this here: https://github.com/deepmind/dsprites-dataset
        dataset_zip = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding='bytes')
        imgs = dataset_zip['imgs']
        imgs = imgs[:, :, :, None] # make into 4d tensor

        # 90% random test/train split
        n_data = len(imgs)
        np.random.shuffle(imgs)
        data_train = imgs[0 : (9*n_data)//10]
        data_test = imgs[(9*n_data)//10 : ]

        return data_train, data_test
    """
    """
    Init model from inputs to outputs
    """
    def _model_init(self):
        with tf.device('/cpu:0'):
            # make placeholder for feeding in data labels during training and evaluation
            input_label = tf.placeholder(shape=None, dtype=tf.float32, name="label")
            #one_hot_labels = tf.one_hot(labels, 2, dtype=tf.float32)
            # make placeholder for feeding in data during training and evaluation
            input_batch = tf.placeholder(shape=[None, 21, 21, 4], dtype=tf.float32, name="input")
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
        net = self._layers_init()       
        #feed-forward input and get logits output.
        logits = net.forward(input_label)
        #pass logits through softmax for network output ([0,1] probability)
        #TODO: add softmax to modules
        output_probabilities = tf.nn.softmax(logits)
        #predicted classes
        output_pred_cls = tf.argmax(output_probabilities, 1)
        #true classes
        #y_true_cls = tf.argmax(y_, 1)

        return input_batch, input_label, output_probabilities, output_pred_cls

    """
    define layers of model
    """
    def _layers_init(self, inputs):
        return sequential32.Sequential([convolution32.Convolution(kernel_size=4,
                                                                  output_depth=32, 
                                                                  input_depth=4,
                                                                  input_dim=27, act ='relu',
                                                                  stride_size=1, pad='VALID'),
                                    
                                        
                           convolution32.Convolution(kernel_size=3, output_depth=32,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                           
                           maxpool32.MaxPool(),
    
                           convolution32.Convolution(kernel_size=3, output_depth=64,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                                        
                           convolution32.Convolution(kernel_size=3, output_depth=64,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                                        
                           convolution32.Convolution(kernel_size=3, output_depth=64,
                                                     stride_size=1, act ='relu',
                                                     pad='SAME'),
    
                           maxpool32.MaxPool(),
                           
                           linear32.Linear(64, act ='relu', keep_prob=self.dropout_prob, 
                                           use_dropout = True),
                                                   
                           linear32.Linear(64, act ='relu', keep_prob=self.dropout_prob, 
                                           use_dropout = True),
                                        
                           avgCyclicPool2.CyclicAvgPool(),
    
                           linear32.Linear(2, act ='linear')])



    """
    def _loss_init(self):
        ### Regulariser part of loss has two parts: KL divergence and Total Correlation
        ## KL part:
        KL_divergence = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(self.enc_logvar) - self.enc_logvar + self.enc_mean**2,axis=1) - self.z_dim)

        ## Total Correlation part:
        # permuted samples from q(z)
        real_samples = self.z_sample
        permuted_rows = []
        for i in range(real_samples.get_shape()[1]):
            permuted_rows.append(tf.random_shuffle(real_samples[:, i]))
        permuted_samples = tf.stack(permuted_rows, axis=1)

        # define discriminator network to distinguish between real and permuted q(z)
        logits_real, probs_real = self._discriminator_init(real_samples)
        logits_permuted, probs_permuted = self._discriminator_init(permuted_samples, reuse=True)

        # FactorVAE paper has gamma * log(D(z) / (1- D(z))) in Algorithm 2, where D(z) is probability of being real
        # Let PT be probability of being true, PF be probability of being false. Then we want log(PT/PF)
        # Since PT = exp(logit_T) / [exp(logit_T) + exp(logit_F)]
        # and  PT = exp(logit_F) / [exp(logit_T) + exp(logit_F)], we have that
        # log(PT/PF) = logit_T - logit_F
        tc_regulariser = self.gamma * tf.reduce_mean(logits_real[:, 0]  - logits_real[:, 1], axis=0)

        total_regulariser = KL_divergence + tc_regulariser

        ### Reconstruction loss is bernoulli
        im = self.input_ph
        im_flat = tf.reshape(im, shape=[-1, 64*64*1])
        logits = self.dec_stoch
        logits_flat = tf.reshape(logits, shape=[-1, 64*64*1])
        recon_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_flat,
                                                    labels=im_flat),
                                                    axis=1),
                                                    name="recon_loss")

        auto_encoder_loss = tf.add(recon_loss, total_regulariser, name="auto_encoder_loss")

        ### Loss for discriminator
        disc_loss = tf.add(0.5 * tf.reduce_mean(tf.log(probs_real[:, 0])), 0.5 * tf.reduce_mean(tf.log(probs_permuted[:, 1])), name="disc_loss")

        return recon_loss, auto_encoder_loss, disc_loss


    def _optimizer_init(self):
        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        ae_train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.auto_encoder_loss, var_list=enc_vars+dec_vars)
        disc_train_step = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(-self.disc_loss, var_list=disc_vars)

        return ae_train_step, disc_train_step
    """

if __name__ == "__main__":
    mode = sys.argv[1]
    vae = FactorVAE()
    if mode == "train":
        vae.train()
    elif mode == "load":
vae.load_latest_checkpoint()