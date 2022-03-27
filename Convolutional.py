import numpy as np
import cv2
import tensorflow as tf
import warnings
from IPython import embed
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


class conv_op():
    """ This class is for applying the convolut
    """
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.rand(num_filters,filter_size, filter_size)/(filter_size*filter_size)
        
    def image_region(self, image):
        height, width = image.shape
        for i in range(height - self.filter_size + 1):
            for j in range(width - self.filter_size +1):
                image_patch = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield image_patch, i , j
                
    def forward_prop(self, image):
        height, width = image.shape
        self.image = image
#         print(self.image)
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for image_patch, i, j in self.image_region(image):
#             print("self.conv_filter.shape", self.conv_filter.shape)
            conv_out[i, j] = np.sum(image_patch*self.conv_filter,axis = (1,2))
        return conv_out
    
    def back_prop(self, DL_dout,learning_rate = 0.005):
#         print("self.conv_filter.shape",self.conv_filter)
#         embed()
        dl_df_params = np.zeros(self.conv_filter.shape)
        for image_patch, i, j in self.image_region(self.image):
            for k in range(self.num_filters):
                dl_df_params[k] += image_patch * DL_dout[i, j, k]
            
        # Filter Params update
        self.conv_filter -= learning_rate*dl_df_params
        return dl_df_params