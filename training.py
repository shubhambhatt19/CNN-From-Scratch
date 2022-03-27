import numpy as np
import cv2
import tensorflow as tf
import warnings
from IPython import embed
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


### User Defined functions####
from Convolutional import conv_op
from Pooling import max_pool
from activation import softmax_cost

# Loading the data from keras
from tensorflow.keras.datasets import mnist
data = mnist.load_data()
(X_train, Y_train), (X_test, Y_test) = data
print(f"Shape of X_train is {X_train.shape}")
print(f"Shape of Y_train is {Y_train.shape}")
print(f"Shape of X_test is {X_test.shape}")
print(f"Shape of Y_test is {Y_test.shape}")


frwd = conv_op(10,3)
max_pooling = max_pool(2)
softmax_out = softmax_cost(13*13*10, 10)


def cnn_forward_prop(image, label):
    out  = frwd.forward_prop((image/255)-0.5)
    
    out2 = max_pooling.forward_prop(out)
    
    out3 = softmax_out.forward_prop(out2)
    
    cross_entropy_loss = -np.log(out3[label])
    accuracy_eval = 1 if np.argmax(out3) == label else 0
    
    return out3, cross_entropy_loss, accuracy_eval


def training(image, label, lr = 0.005):
    
    out, loss, acc = cnn_forward_prop(image, label)
    
    gradient = np.zeros(10)
    
    gradient[label] = -1/out[label]
    
    #backprop
    grad_back = softmax_out.back_prop(gradient,lr)
    grad_back = max_pooling.back_prop(grad_back)
    grad_back = frwd.back_prop(grad_back,lr)
    
    return loss, acc


step_plot = []
loss_plot = []
plt.figure(figsize = (10, 10));

for epoc in range(50):
    loss = 0
    num_correct = 0
    
    for i , (im,label) in enumerate(zip(X_train,Y_train)):
        if i % 100 == 0:
            print("Steps: %d, Loss: %.3f and Accuracy: %d%%" %(i+1,loss/100, num_correct))
            step_plot.append(str(epoc)+"-"+str(i))
            loss_plot.append(loss/100)
            loss = 0
            num_correct = 0
        l1, acc = training(im, label)
        loss += l1
        num_correct += acc
        
plt.plot(step_plot, loss_plot);
plt.show();
