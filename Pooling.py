import numpy as np
import cv2
import tensorflow as tf
import warnings
from IPython import embed
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


class max_pool():
    def __init__(self, filter_size):
        self.filter_size = filter_size
        
    def image_region(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width = image.shape[1] // self.filter_size
        self.image = image
        
        for i in range(new_height - self.filter_size + 1):
            for j in range(new_width - self.filter_size +1):
                image_patch = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield image_patch, i , j
                
    def forward_prop(self, image):
        height, width,num_filters = image.shape
        conv_out = np.zeros((height // self.filter_size , width // self.filter_size , num_filters))
        for image_patch, i, j in self.image_region(image):
            conv_out[i, j] = np.amax(image_patch)
        return conv_out
    
    def back_prop(self,dl_dout):
        """dl_dout is coming from softmax"""
        dl_dout_pool = np.zeros(self.image.shape)        
        for image_patch, i, j in self.image_region(self.image):
            height, width , num_filters = image_patch.shape
            for i in range(height):
                for j in range(width):
                    for k in range(num_filters):
#                         print("image_patch[i, j, k]",image_patch[i, j, k],"np.amax(image_patch , axis = (0,1))", np.amax(image_patch , axis = (0,1)))
                        if image_patch[i, j, k] == np.amax(image_patch):
                            dl_dout_pool[i*self.filter_size+i , j*self.filter_size + j, k] = dl_dout_pool[i, j, k]
            return dl_dout_pool