import numpy as np
from utils import load_dataset
from utils import convert_to_one_hot
from layers.convolutional_layer import Conv
from layers.fullyconnected import FullyConnected
from layers.flatten import Flatten
from layers.max_pool import MaxPool
from activations import relu, lkrelu, linear, sigmoid, cross_entropy
from neural_network import Network

(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes) = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))

layers = [
        Conv((5, 5, 3, 8), strides=1,pad=2, activation=relu, filter_init=lambda shp: np.random.normal(size=shp) * 1.0 / (5*5*3)),
        MaxPool(f=8, strides=8, channels = 8),
        Conv((3, 3, 8, 16), strides=1,pad=1, activation=relu, filter_init=lambda shp:  np.random.normal(size=shp) * 1.0 / (3*3*8)),
        MaxPool(f=4, strides=4, channels = 16),
        Flatten((2, 2, 16)),
        FullyConnected((2*2*16, 20), activation=sigmoid, weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (2*2*16 + 20))),
        FullyConnected((20, 6), activation=linear, weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / ( 20+ 6)))
]

lr = 0.001
k = 2000
net = Network(layers, lr=lr, loss=cross_entropy)

for epoch in xrange(10000):
    net.train_step((X_train[:10], Y_train[:10]))
    loss = np.sum(cross_entropy.compute((net.forward(X_train[:10]), Y_train[:10])))