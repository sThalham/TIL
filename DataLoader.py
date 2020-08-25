#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:22:42 2019

@author: marcop
"""

from glob import glob
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
# from PIL import Image
import imageio
import transforms3d
import json
import os
import open3d
from pose_error import add, adi
from sklearn.preprocessing import minmax_scale
import random
#from scipy.spatial.transform import Rotation as R
import math
import gc



class DataLoader():
    
    def __init__(self, model_nr, img_res=(64,64)):
        
       
        self.img_res = img_res
        self.model_nr = model_nr
        self.train_path = os.path.join(os.path.normpath(r'.\Data\3D\train_data'), f'{model_nr:06}') 
        self.test_path = os.path.join(os.path.normpath(r'.\Data\3D\test_data'), f'{model_nr:06}') 

        self.annotations = self.load_json("image_data")[str(model_nr)]
        self.canonical_pose = self.load_json("image_data")["canonical"][str(model_nr)]
        self.model_data = self.load_json("model_data")[str(model_nr)]

        self.error_function = adi if model_nr in [10,11] else add
        
        
        
    def load_json(self,filename):
        with open(filename + '.json') as json_file:
            data = json.load(json_file)
        return data
        
    
    def load_img(self, img_path):
         image =  imageio.imread(img_path)
         return np.array(image)
        
    def load_batch(self, batch_size = 1, train = True):
        if train:
            p = self.train_path  
        else:
            p = self.test_path
            
        path = glob(p + r"\*")
        self.n_batches = int(len(path) / batch_size)
        total_samples = self.n_batches * batch_size
        
        # shuffle pictures
        path = np.random.choice(path, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            inputs = []
            anns = []

            for img in batch:
                
                image_nr = img.split("\\")[-1].split(".")[0]
                output_anno = self.getImageAnnotation(image_nr)
                
                input_anno = self.canonical_pose
                input_nr = input_anno["img_path"].split("\\")[-1].split(".")[0]
                
                augmented_img = self.augment_data(input_anno["img_path"], input_anno, train = True)
                inputs.append(np.expand_dims(augmented_img[:,:,0:3],0))
                
                rot = output_anno['Rot']
                inputs.append(np.expand_dims(rot,0))
                
                x,y,width,height = [0 if i < 0 else i for i in output_anno["obj_bb"]]
                        
                img_output = self.load_img(output_anno["img_path"])
                img_output = cv2.resize(img_output, self.img_res, interpolation=cv2.INTER_CUBIC)
                # img_output = cv2.resize(img_output[y:y+height, x:x+width, :], self.img_res, interpolation=cv2.INTER_CUBIC)

                img_output = img_output / 255
                anns.append(np.expand_dims(img_output,0))

                # anns.append(np.expand_dims(img_c, 0))
                # inputs = inputs
                # anns = np.array(anns)
                
            yield inputs, anns


    def getImageAnnotation(self, image_nr):
        return self.annotations[int(image_nr)]   
  
    def augment_data(self, img_path, img_data, train = True):
        img_res = self.img_res
        img = self.load_img(img_path)
        img_seq= np.expand_dims(img,0)
        if train: # only augment training images
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            
            seq =  iaa.Sequential(
                [
                  iaa.AdditiveGaussianNoise(loc=0, scale=(0, 1.5), per_channel=0.5), # add gaussian noise to images

                ],
                random_order=True)
            
            img_seq = seq(images=img_seq)        
            ####################
        
        img = img_seq[0]
           
        # resize
        img = cv2.resize(img, img_res,interpolation=cv2.INTER_CUBIC)

        img_standardized = img/255#self.standardize(img)
        return img_standardized
        
    
    def standardize(self,img):
        img = img.astype('float32')
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        img = np.clip(img, -1.0, 1.0)

        img = img/ 2.0
        
        
        return img
   
        