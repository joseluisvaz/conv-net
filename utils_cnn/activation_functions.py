import numpy as np

def RELU_Layer(x_prev):
    return np.maximum(0, x_prev)

def RELU_layer_backward(x):
    return (x > 0).astype(int)