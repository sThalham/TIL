#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:32:47 2019

@author: marcop
"""

from DataLoader import DataLoader
from NeuralNet import NeuralNet
import datetime

import gc

from keras import backend as K


def calc_lr(start, epoch):
    if epoch < 30:
        return start
    else:
        return start / 10
    

objects = [4]
#model_nr = 14
batch_size = 1
epochs = 10

# import tensorflow.compat.v1 as tf
shapes = [(8,8)]
losses = ["mse", "mae" , "SSIM", "SSIM_MAE"]
losses = ["mse"]
for model_nr in objects:
    l = DataLoader(model_nr)
    
    for loss_function in losses:
        if model_nr == 6 and loss_function in ["mse", "mae"]:
            continue
    
        for shape in shapes:
            n = NeuralNet(loss_function, shape, model_name = r".\snapshots\04\mse_featuremapsize8_epoch_30")
            # n = NeuralNet(loss_function, shape)
           
            epochs_already_trained = 30
            
            start_time = datetime.datetime.now()
            
            for epoch in range(1,epochs+1):
                ##### insert learning rate decay
                
                current_learning_rate = calc_lr(1e-3, epoch)
                K.set_value(n.model.optimizer.lr, current_learning_rate)  # set new lr
                for batch_i, (images, anns) in enumerate(l.load_batch(batch_size = batch_size)):
                      
                        loss = n.model.train_on_batch(images, anns)
                    
                        if not batch_i%50:
                            # Plot the progress
                            print("Object %d [Epoch %d/%d] [Batch %d/%d] [loss_img: %f] [loss_function: %s] " \
                                % (model_nr, epoch, epochs,
                                batch_i, l.n_batches,
                                loss, loss_function))
                        
                                                       
                # for batch_i, (images, anns) in enumerate(l.load_batch(batch_size = batch_size, train = False)):
                #         predictions = n.model.predict_on_batch(images)
                #         # counter += 1
                #         # correct_elements += l.postprocessing(predictions[0],anns[0])
                        
                #         loss = n.model.test_on_batch(images,anns)
                #         # Plot the progress
                #         # print("Test:  Object %d [Epoch %d/%d]  [correct_elements: %d/%d], pred: %f, ann: %f" \
                #         #       % (model_nr, epoch, epochs, correct_elements,counter, predictions[0], anns[0]))
                #         print("Test:  Object %d [Epoch %d/%d] number: %d/%d loss: %f" \
                #               % (model_nr, epoch, epochs, batch_i, l.n_batches, loss))
        
                gc.collect()
                if not epoch%10:
                    n.save_snapshot(model_nr, epoch + epochs_already_trained)

