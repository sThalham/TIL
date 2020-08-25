# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:59:05 2019

@author: marco
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.layers import Lambda

if K.backend() == 'tensorflow':
    import tensorflow as tf
    import math
    
    def K_meshgrid(x, y, z):
        return tf.meshgrid(x, y, z)

    def K_linspace(start, stop, num):
        return tf.linspace(start, stop, num)

else:
    raise Exception("Only 'tensorflow' is supported as backend")


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]

        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):         
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, depth, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')
        z = K.cast(K.flatten(sampled_grids[:, 2:3, :]), dtype='float32')

        # bring into [0,1*shape]
        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')
        z = .5 * (z + 1.0) * K.cast(depth, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1
        z0 = K.cast(z, 'int32')
        z1 = z0 + 1
        
        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)
        max_z = int(depth - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)
        z0 = K.clip(z0, 0, max_z)
        z1 = K.clip(z1, 0, max_z)
       
        indices_a  = K.transpose(K.reshape(K.flatten([[y0,x0,z0]]), (3,-1)))
        indices_b =  K.transpose(K.reshape(K.flatten([[y1,x0,z0]]), (3,-1)))
        indices_c =  K.transpose(K.reshape(K.flatten([[y0,x1,z0]]), (3,-1)))
        indices_d =  K.transpose(K.reshape(K.flatten([[y1,x1,z0]]), (3,-1)))
        
        indices_e =  K.transpose(K.reshape(K.flatten([[y0,x0,z1]]), (3,-1)))
        indices_f =  K.transpose(K.reshape(K.flatten([[y1,x0,z1]]), (3,-1)))
        indices_g =  K.transpose(K.reshape(K.flatten([[y0,x1,z1]]), (3,-1)))
        indices_h =  K.transpose(K.reshape(K.flatten([[y1,x1,z1]]), (3,-1)))
              
        # get 3 dimensional image -> 2d image in center of 3d, zeros around
        my_img = K.reshape(image, ( height, width, 1, num_channels))
        tf_zeros = tf.zeros((       height, width, int((depth-1)/2), num_channels))
        d_image = tf.concat([tf_zeros, my_img, tf_zeros], 2)
        
        
        pixel_values_a = tf.gather_nd(d_image, indices_a)
        pixel_values_b = tf.gather_nd(d_image, indices_b)
        pixel_values_c = tf.gather_nd(d_image, indices_c)
        pixel_values_d = tf.gather_nd(d_image, indices_d)
        
        pixel_values_e = tf.gather_nd(d_image, indices_e)
        pixel_values_f = tf.gather_nd(d_image, indices_f)
        pixel_values_g = tf.gather_nd(d_image, indices_g)
        pixel_values_h = tf.gather_nd(d_image, indices_h)
      
        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')
        z0 = K.cast(z0, 'float32')
        z1 = K.cast(z1, 'float32')
       
        # V_a = K.expand_dims(((x1 - x) * (y1 - y) * (z1 - z)), 1) # x0 y0 z0
        # V_b = K.expand_dims(((x1 - x) * (y - y0) * (z1 - z)), 1)  # x0 y1 z0
        # V_c = K.expand_dims(((x - x0) * (y1 - y) * (z1 - z)), 1)
        # V_d = K.expand_dims(((x - x0) * (y - y0) * (z1 - z)), 1)
        
        # V_e = K.expand_dims(((x1 - x) * (y1 - y) * (z - z0)), 1) # x0 y0 z1
        # V_f = K.expand_dims(((x1 - x) * (y - y0) * (z - z0)), 1) 
        # V_g = K.expand_dims(((x - x0) * (y1 - y) * (z - z0)), 1)
        # V_h = K.expand_dims(((x - x0) * (y - y0) * (z - z0)), 1)
        
        V_a = K.expand_dims((x - x0) * (y - y0) * (z - z0), 1) # x0 y0 z0
        V_c = K.expand_dims((x - x0) * (y1 - y) * (z - z0), 1)  # x0 y1 z0
        V_b = K.expand_dims((x1 - x) * (y - y0) * (z - z0), 1)
        V_d = K.expand_dims((x1 - x) * (y1 - y) * (z - z0), 1)
        
        V_e = K.expand_dims((x - x0) * (y - y0) * (z1 - z), 1) # x0 y0 z1
        V_g = K.expand_dims((x - x0) * (y1 - y) * (z1 - z), 1) 
        V_f = K.expand_dims((x1 - x) * (y - y0) * (z1 - z), 1)
        V_h = K.expand_dims((x1 - x) * (y1 - y) * (z1 - z), 1)
        
        # area_h = K.expand_dims(((x1 - x) * (y1 - y) * (z1 - z)), 1) # x0 y0 z0
        # area_g = K.expand_dims(((x1 - x) * (y - y0) * (z1 - z)), 1)  # x0 y1 z0
        # area_f = K.expand_dims(((x - x0) * (y1 - y) * (z1 - z)), 1)
        # area_e = K.expand_dims(((x - x0) * (y - y0) * (z1 - z)), 1)
        
        # area_d = K.expand_dims(((x1 - x) * (y1 - y) * (z - z0)), 1) # x0 y0 z1
        # area_c = K.expand_dims(((x1 - x) * (y - y0) * (z - z0)), 1) 
        # area_b = K.expand_dims(((x - x0) * (y1 - y) * (z - z0)), 1)
        # area_a = K.expand_dims(((x - x0) * (y - y0) * (z - z0)), 1)
        
        values_a = V_a * pixel_values_h
        values_b = V_b * pixel_values_g
        values_c = V_c * pixel_values_f
        values_d = V_d * pixel_values_e
        
        values_e = V_e * pixel_values_d
        values_f = V_f * pixel_values_c
        values_g = V_g * pixel_values_b
        values_h = V_h * pixel_values_a
        
        # interpolate in 2D
        # area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        # area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        # area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        # area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        # values_a = area_a * pixel_values_a
        # values_b = area_b * pixel_values_b
        # values_c = area_c * pixel_values_c
        # values_d = area_d * pixel_values_d
        
        
        # # return values_a + values_b + values_c + values_d
        # zd = (K.cast(z, 'float32') - z0) / (z1-K.cast(z0, 'float32'))

        # # zd = 1
        # i_z0 = (values_a + values_b + values_c + values_d) *  (1-zd)
        # i_z1 = (values_e + values_f + values_g + values_h) * zd

        return  ((values_a + values_b + values_c + values_d)  +  (values_e + values_f + values_g + values_h))


    def _make_regular_grids(self, batch_size, depth, height, width, channels):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        z_linspace = K_linspace(-1., 1., depth)
        
        x_coordinates, y_coordinates, z_coordinates = K_meshgrid(x_linspace, y_linspace, z_linspace)
        
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        z_coordinates = K.flatten(z_coordinates)
        
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, z_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 4,  height * width * depth))


    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        depth = 65
        transformations = K.reshape(affine_transformation,
                                    shape=(batch_size, 3, 4))
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, depth, *output_size, num_channels)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        
        #return sampled_grids
                
        interpolated_image = self._interpolate(X, sampled_grids, depth, output_size)
        interpolated_image = K.reshape(interpolated_image,  (batch_size, output_size[0], output_size[1], depth, num_channels))
        interpolated_image = tf.math.reduce_sum(interpolated_image, axis = 3) 
           
        return interpolated_image