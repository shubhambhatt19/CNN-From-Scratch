import numpy as np
import cv2
import tensorflow as tf
import warnings
from IPython import embed
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


class softmax_cost():
    
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.rand(input_node, softmax_node)/input_node
        self.bias = np.zeros(softmax_node)
        
    def forward_prop(self, image):
        self.orig_im_shape = image.shape
        flatted_image = image.flatten()
        self.modified_input = flatted_image
#         embed()
        out_val = np.dot(flatted_image,self.weight)+self.bias
        self.out = out_val
        eout = np.exp(out_val, dtype = np.float)
        return eout/np.sum(eout,axis = 0)
    
    def back_prop(self, dl_out, learning_rate):
        for i , grad in enumerate(dl_out):
            if grad == 0:
                continue
            
            #against total
            transformation_eq = np.exp(self.out)
            s_total = np.sum(transformation_eq)
            
            #gradients with respect to out
            dy_dz = -transformation_eq[i]*transformation_eq/(s_total ** 2)
            dy_dz[i] = transformation_eq[i]*(s_total - transformation_eq[i])/(s_total ** 2)
            
            # gradient wrt to totals against weights/biases/input
            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight
            
            dl_dz = grad * dy_dz
            
            dl_dw = dz_dw[np.newaxis].T @ dl_dz[np.newaxis]
            dl_db = dl_dz * dz_db
            
            dl_d_inp = dz_d_inp @ dl_dz
            
            self.weight -= learning_rate * dl_dw
            self.bias -= learning_rate * dl_db
            
            return dl_d_inp.reshape(self.orig_im_shape)