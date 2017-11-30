import numpy as np
from utils import load_dataset
from utils import convert_to_one_hot
from utils import accuracy
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

batch_size = 20

lr = 0.0001
k = 2000
net = Network(layers, lr=lr, loss=cross_entropy)

for epoch in xrange(10000):
    shuffled_index = np.random.permutation(X_train.shape[0])
    batch_train_X = X_train[shuffled_index[:batch_size]]
    batch_train_Y = Y_train[shuffled_index[:batch_size]]
    net.train_step((batch_train_X, batch_train_Y))
    loss = np.sum(cross_entropy.compute((net.forward(X_train[:10]), Y_train[:10])))
    print 'Epoch: %d loss : %f' % (epoch, loss)
    if epoch % 1000 == 1:
        print 'Accuracy on first 50 test set\'s batch : %f' % accuracy(net, X_test[:50], Y_test[:50])
    if epoch % 5000 == 5000 - 1:
        print 'Accuracy over all test set %f' % accuracy(net, X_test, Y_test)