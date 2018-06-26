#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train Enhanced DeepHits just like in jupyetnotebook; sequential script
@author: ereyes
"""

#python 2 and 3 comptibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#enables acces to parent folder ()
import os
import sys
sys.path.append("/home/ereyes/Alerce/AlerceDHtest/modules")

#other imports
import tensorflow as tf
import numpy as np

#model and dataset
from hits2013 import DH_set
from Enhanced_DeepHiTS import DeepHiTS

#%%
#number of samples for every trainning iteration
BATCH_SIZE = 50
#First 100k iterations of trainning
INITIAL_PATIENCE = 100000  
#Validate trainning every 10k iterations
VALIDATION_PERIOD = 10000  
#Update learning rate every 100k iterations
ANNEALING_PERIOD = 100000  
#If validation criteria is met, increase trainning iterations by 100k
PATIENCE_INCREMENT = 100000
#path where summaries, graph and best  trained models will be saved
SUMMARY_DIR = "TB/cap_runs/"
#path to *.tfrecord files (data) in your computer
data_path = '/home/ereyes/LRPpaper/datasets'

#%%
#define dataset
DH_data = DH_set(data_path= data_path, BATCH_SIZE=BATCH_SIZE)
#define model
DH = DeepHiTS()

y_pred_cls = DH.output_pred_cls
y_true_cls = DH.input_label
cost = DH.loss
sess = DH.sess
validation_images = DH_data.validation_images
validation_labels = DH_data.validation_labels
test_images = DH_data.test_images
test_labels = DH_data.test_labels
train_images = DH_data.train_images
train_labels = DH_data.train_labels
keep_prob = DH.dropout_prob
images = DH.input_batch
labels = DH.input_label
learning_rate = DH.learning_rate
train_op = DH.train_step

#%%
#accuracy measurement
with tf.name_scope('accuracy'):  
        #compare predictions
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        #get accuracy
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
#tensorflow summary        
accuracy_sum = tf.summary.scalar('accuracy', accuracy)

#metrics to be used on validation criteria
metrics = (cost, accuracy)
#names for those metrics
metrics_names = ('xentropy', 'accuracy')
#list of metrics as a TF summary
metric_summary_list = []

#add metrics to summary list
for metric, name in zip(metrics, metrics_names):
    summary = tf.summary.scalar(name, metric)
    metric_summary_list.append(summary)

#get unique summary for validation criteria metrics
merged_summaries = tf.summary.merge(metric_summary_list)

#%%
#other measurements
with tf.name_scope('performance_measures'):
    #confusion matrix values
    with tf.name_scope('values'):
        #true positives
        TP = tf.count_nonzero((y_pred_cls * y_true_cls))
        #true negative
        TN = tf.count_nonzero((y_pred_cls - 1) * (y_true_cls- 1))
        #false positives
        FP = tf.count_nonzero(y_pred_cls * (y_true_cls - 1))
        #false negatives
        FN = tf.count_nonzero((y_pred_cls - 1) * y_true_cls)
        
    with tf.name_scope('accuracy_func'):
            acc_mes = (TP+TN)/(TP+TN+FN+FP)
        
    with tf.name_scope('precision_func'):
            prec_mes = TP/(TP+FP)
            
    with tf.name_scope('recall_func'):
            rec_mes = TP/(TP+FN)      
            
    with tf.name_scope('f1_func'):
            f1_mes = 2 * prec_mes * rec_mes / (prec_mes + rec_mes) 
            
#%%
#TF writers and saver of model
train_writer = tf.summary.FileWriter(os.path.join(SUMMARY_DIR, 'train'),
                                     sess.graph,
                                     max_queue=1000)
validation_writer = tf.summary.FileWriter(os.path.join(SUMMARY_DIR, 'validation'))

saver = tf.train.Saver()

#%%
#Validation Function
"""
Called by trainning functiontion to validate model, it gets accuracy and loss
of model in validation set (100k samples), save best model so far and verify validation criteria. 
If criteria is met, another PATIENCE_INCREMENT iterations of trainning will be performed.

@param current_iteration
@param current_patience trainning iterations left 
@param best_model accuracy and loss of best model so far
@param stopping_criteria_model loss and accuracy of last model that met validation criteria
@return current_patience or new_patience (current_iterations+PATIENCE_INCREMENT) if validation criteria
is met 
"""
def validate(current_iteration, current_patience, best_model, stopping_criteria_model):
    #check if current_iteration is multiple of VALIDATION_PERIOD
    #to perform validation every VALIDATION_PERIOD iterations
    if current_iteration % VALIDATION_PERIOD != 0:
        return current_patience
    
    #to store validation accuracy and loss
    metric_data = {}
    for metric in metrics:
        metric_data[metric] = {
            'values_per_batch': [],
            'batch_mean': None
        }

    #Validate validation set.
    for val_batch in range(100000 // BATCH_SIZE):
        #get validation batch from tfrecords
        images_array, labels_array = sess.run((
            validation_images,
            validation_labels))
        #get accuracy and loss
        metrics_value = sess.run(metrics,
                                 feed_dict={
                                     keep_prob: 1.0,
                                     images: images_array,
                                     labels: labels_array
                                 })
        #append every batch metric to metric_data
        for metric, value in zip(metrics, metrics_value):
            metric_data[metric]['values_per_batch'].append(value)
            
    #get accuracy and loss for al validation batches and write them to validation writer
    #as a summary
    for metric, name in zip(metrics, metrics_names):
        metric_data[metric]['batch_mean'] = np.array(metric_data[metric]['values_per_batch']).mean()
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=float(metric_data[metric]['batch_mean']))
        validation_writer.add_summary(summary, current_iteration)
    
    #mean accuracy of validation set
    accuracy_mean = metric_data[accuracy]['batch_mean']
    #Check if accuracy is over best model so far and overwrite model checkpoint   
    if accuracy_mean > best_model['accuracy']:
        best_model['accuracy'] = accuracy_mean
        best_model['iteration'] = current_iteration
        print("New best model: Accuracy %.4f @ it %d" % (
            best_model['accuracy'],
            best_model['iteration']
        ))
        #ckpt_dir = os.path.join(SUMMARY_DIR, 'ckpt_files')
        #if not os.path.exists(ckpt_dir):
        #    os.makedirs(ckpt_dir)
        saver.save(sess, SUMMARY_DIR)

    #check stopping criteria, which is:
    #if current model error is under 99% of the error of 
    #last model that met stopping criteria, validation criteria is met
    if (1.0-accuracy_mean) < 0.99*(1.0-stopping_criteria_model['accuracy']):
        stopping_criteria_model['accuracy'] = accuracy_mean
        stopping_criteria_model['iteration'] = current_iteration
        new_patience = current_iteration + PATIENCE_INCREMENT
        if new_patience > current_patience:
            print("Patience increased to %d because of model with accuracy %.4f @ it %d" % (
                new_patience,
                stopping_criteria_model['accuracy'],
                stopping_criteria_model['iteration']
            ))
            return new_patience
        else:
            return current_patience
    else:
        return current_patience
    
#%%
#test function
"""
Evaluate model on specific set.

@param set_images
@param set_labels
@return loss, accuracy, precision, reacall and f1_score of model
"""
def eval_set(set_images, set_labels):
    #create list of metric to store per batch evaluation
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    #get metrics on every batch
    for test_batch in range(100000 // BATCH_SIZE):
        images_array, labels_array = sess.run((
            set_images,
            set_labels))
        loss_val, acc_val, prec, rec, f1 = sess.run((cost, acc_mes, prec_mes, rec_mes, f1_mes),
                                     feed_dict={
                                         keep_prob: 1.0,
                                         images: images_array,
                                         labels: labels_array
                                     })
        losses.append(loss_val)
        accuracies.append(acc_val)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    #average all batch metric on a single one
    loss_mean = np.array(losses).mean()
    accuracy_mean = np.array(accuracies).mean()
    precision_mean = np.array(precisions).mean()
    recall_mean = np.array(recalls).mean()
    f1_mean = np.array(f1s).mean()
    

    return loss_mean, accuracy_mean, precision_mean, recall_mean, f1_mean

"""
Evaluate model on test set (100k samples).

@return loss, accuracy, precision, reacall and f1_score of model ofer test
"""
def test():
    return eval_set(test_images, test_labels)

#%%
#Learning rate update and trainning  functions
def update_learning_rate(global_step):
    sess.run(tf.assign(learning_rate,
                       0.04/(2.0**(global_step//ANNEALING_PERIOD))))
    lr_value = sess.run(learning_rate)
    print("Iteration %d. Learning rate: %.4f" % (global_step, lr_value))

"""
Train model from scratch, it means to reinitialice model parameters (weights and biases), 
set current iteration to 0 (global_step), and initialize best_model and stopping_criteria_model
to 0.5 accuracy. After trainning process restores best model and it perform an evaluation over test set.


@return test_accuracy, test_prec, test_rec, test_f1 of best trainning model
"""
def train_from_scratch():
    #initialize params
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init_op)
    DH._variables_init()
    #init current it to 0
    global_step = 0
    patience = INITIAL_PATIENCE
    #init models acc to 0.5
    best_model = {
        'iteration': 0,
        'accuracy': 0.5
    }
    stopping_criteria_model = {
        'iteration': 0,
        'accuracy': 0.5
    }
    #train model
    while global_step < patience:
        #check if learning rate must be updated
        if global_step % ANNEALING_PERIOD == 0:
            update_learning_rate(global_step)
        #get train samples
        images_array, labels_array = sess.run((
            train_images,
            train_labels))
        #perform a trainning iteration
        summaries_output, _ = sess.run((merged_summaries,
                                        train_op),
                                       feed_dict={
                                           keep_prob: 0.5,
                                           images: images_array,
                                           labels: labels_array
                                       })
        global_step += 1
        #trainning writer is commented due large amount of data that it stores
        #write accuracy and loss summaries to train writer
        #train_writer.add_summary(summaries_output, global_step)
        
        #validate model after trainning iteration
        patience = validate(global_step, patience, best_model, stopping_criteria_model)
    
    #restore best model so far
    saver.restore(sess, SUMMARY_DIR)
    #evaluate model over test set
    test_loss, test_accuracy, test_prec, test_rec, test_f1 = test()
    print("Best model @ it %d.\nValidation accuracy %.5f, Test accuracy %.5f" % (
        best_model['iteration'],
        best_model['accuracy'],
        test_accuracy
    ))
    print("Test loss %.5f" % test_loss)
    
    return test_accuracy, test_prec, test_rec, test_f1

#%%
#TRAIN MODEL
#Numbers of model to train
N_models = 1

#list of test set parameters to save of each model
acc_ls = []
prec_ls = []
rec_ls = []
f1_ls = []

#list of validation set parameters to save of each model
acc_ls_val = []
prec_ls_val = []
rec_ls_val = []
f1_ls_val = []


for i in range(N_models):
        
    print("\nModel:", i)
    test_accuracy, test_prec, test_rec, test_f1 = train_from_scratch()
        
    acc_ls.append(test_accuracy)
    prec_ls.append(test_prec)
    rec_ls.append(test_rec)
    f1_ls.append(test_f1)
    
    _, val_accuracy, val_prec, val_rec, val_f1 = eval_set(validation_images, validation_labels)
    
    acc_ls_val.append(val_accuracy)
    prec_ls_val.append(val_prec)
    rec_ls_val.append(val_rec)
    f1_ls_val.append(val_f1)
    
#%%
#Print metrics
"""
get mean and std of a list of models parameters

@metric_ls list of metrics
@metric_name
@return metric_mean
@return metric_std
"""
def getMetricVal(metric_ls, metric_name):
    metric_mean = np.array(metric_ls).mean()*100
    metric_std = np.array(metric_ls).std()*100
    print("%s %.2f +/- %.2f" % (
        metric_name,
        np.array(metric_ls).mean()*100,
        np.array(metric_ls).std()*100
    ))
    return metric_mean, metric_std

print("Test Metrics\n")
_,_=getMetricVal(acc_ls, 'Accuracy')

_,_=getMetricVal(prec_ls, 'Precision')

_,_=getMetricVal(rec_ls, 'Recall')

_,_=getMetricVal(f1_ls, 'F1 Score')

print("Val Metrics\n")
_,_=getMetricVal(acc_ls_val, 'Accuracy')

_,_=getMetricVal(prec_ls_val, 'Precision')

_,_=getMetricVal(rec_ls_val, 'Recall')

_,_=getMetricVal(f1_ls_val, 'F1 Score')