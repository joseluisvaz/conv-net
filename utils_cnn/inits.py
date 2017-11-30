import numpy as np

def uniform_init(W_shape):
    """
    Returns a matrix with shape (W_shape) with samples from the uniform distribution (0,1)
    :param W_shape:
    :return: W_initialized
    """
    W_initialized = np.random.rand(*W_shape)
    return W_initialized

def normal_initialization(W_shape):
    """
    Returns a matrix with shape (W_shape) with samples from the unit normal distribution N(0,1)
    :param W_shape:
    :return: W_initialized
    """
    W_initialized = np.random.randn(*W_shape)
    return W_initialized

def he_initialization(W_shape):
    """
    Returns a matrix with shape (W_shape) with samples from the normal distribution N(0, 2/n_in)
    where n_in is the number of inputs of the specific layer.
    :param W_shape:
    :return: W_initialized
    """
    n_in = 1
    for i in range(len(W_shape)-1):
        n_in *= W_shape[i]
    W_initialized = np.random.randn(*W_shape)*2/n_in
    return W_initialized


