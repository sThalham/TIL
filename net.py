#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:44:16 2019

@author: marcop
"""
import os
import numpy as np
from datetime import datetime

import tensorflow as tf



from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda, MaxPooling2D
from keras.layers import BatchNormalization, Add, ZeroPadding2D, Activation, Softmax
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.activations import relu, linear

from keras.losses import huber_loss

from keras.engine.topology import Layer
from keras.models import load_model
from keras.initializers import RandomNormal
import keras.initializers 
import keras.backend as K
from stn import spatial_transformer_network as transformer
import math
from DataLoader import DataLoader
import keras_resnet.models

from classification_models.keras import Classifiers
from STN_3D import BilinearInterpolation

# import tensorflow_graphics.geometry.transformation as tfg_transf


class NeuralNet():
     
    
    def __init__(self, loss_function, shape = (2,2), model_name = None):
       
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.fv_shape = shape
        
        
        def SSIM_MAE_loss(y_true, y_pred):
           weighting = 0.85
           return  weighting * 0.5 * (1 - tf.reduce_mean(tf.compat.v2.image.ssim(y_true, y_pred, 2))) + (1-weighting) * abs(y_true - y_pred)
       
        def SSIM_loss(y_true, y_pred):
           return  0.5 * (1 - tf.reduce_mean(tf.compat.v2.image.ssim(y_true, y_pred, 2))) 
        
        losses = {
            "mae": "mae",
            "mse": "mse",
            "SSIM": SSIM_loss,
            "SSIM_MAE": SSIM_MAE_loss}
                    
        self.loss_function = losses[loss_function]
        self.loss_str = loss_function

        init = keras.initializers.glorot_normal() #RandomNormal(mean=0.0, stddev=0.1, seed=None)
        
        # define inputs
        i0 = Input(shape=self.img_shape, name = "image")
        i1 = Input(shape=(12,), name = "transformation")
        
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        preprocess_input(i0)
        res_net = ResNet18(input_shape=(64,64,3), include_top=False, weights='imagenet')
        # res_net.summary()
        # for i in res_net.layers:
        #     i.trainable = False
        
        # fv = res_net.layers[64].output
        
        
        def getLayers(res_net, shape, i1):
            if shape == (2,2):
                fv = res_net.output
            if shape == (4,4):
                fv = res_net.layers[65].output
            if shape == (8,8):
                fv = res_net.layers[46].output

            fv_tf = BilinearInterpolation(shape)([fv, i1])
            if shape == (2,2):
                fv_tf = Conv2DTranspose(256, (3,3), kernel_initializer = init, name="dconv0")(fv_tf)
                fv_tf = Activation(relu)(fv_tf)        

            # if shape[0] < 4:
            #    up1 = Conv2DTranspose(256, (3,3), kernel_initializer = init, name="dec_s11")(fv_tf)
            #    up1 = Activation(relu)(up1)
            #    fv_tf = up1
                
            # if shape[0] < 8:
            #    up2 = Conv2DTranspose(128, (3,3), kernel_initializer = init, name="dec_s21")(fv_tf)
            #    up2 = Activation(relu)(up2)        
            #    up2 = Conv2DTranspose(128, (3,3), kernel_initializer = init, name="dec_s22")(up2)
            #    up2 = Activation(relu)(up2)
            #    up2 = BatchNormalization()(up2)
        
            #    fv_tf = up2 
               
            return fv_tf
               
             
        fv_tf = getLayers(res_net, self.fv_shape, i1)

        u = fv_tf
        if self.fv_shape == (2,2):        
            u = Conv2DTranspose(256, (3,3), kernel_initializer = init, name="dconv1")(fv_tf)
            u = Activation(relu)(u)
            
            u = UpSampling2D((2,2))(u)
        
        if self.fv_shape == (8,8):
            u = Conv2DTranspose(256, (5,5), kernel_initializer = init, name="dconv1")(fv_tf)
            u = Activation(relu)(u)
        
        u = Conv2DTranspose(128, (3,3), kernel_initializer = init, name="dconv2")(u)
        u = Activation(relu)(u)

        u = UpSampling2D((2,2))(u)
        
        u = Conv2DTranspose(64, (3,3), kernel_initializer = init, name="dconv3")(u)
        u = Activation(relu)(u)

        u = UpSampling2D((2,2))(u)
        
        u = Conv2DTranspose(3, (5,5), kernel_initializer = init, name="dconv4")(u)
        u = Activation(linear)(u)

        out = u
        
        self.model = Model(inputs=[res_net.input, i1], outputs=[out])
        # self.model.summary()
        
        if model_name:
            self.model = load_model(model_name, custom_objects={'BilinearInterpolation': BilinearInterpolation, 'SSIM_loss' : SSIM_loss, 'SSIM_MAE_loss': SSIM_MAE_loss})  

        # compile model 
        # optimizer = SGD(learning_rate=0.001, clipnorm=1000)
        optimizer = Adam(learning_rate = 1e-3)
        self.model.compile(optimizer=optimizer, loss=[self.loss_function], loss_weights = [100])

      



    def save_snapshot(self, model_nr, epoch):
        path = 'snapshots/' + f'{model_nr:02}'
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        
        self.model.save(path + "/" + str(self.loss_str) + "_" + "featuremapsize" + str(self.fv_shape[0]) + "_epoch_" + str(epoch))
        