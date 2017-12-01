import numpy as np
from abstract_layer import AbstractLayer
from utils import zero_pad

class Conv(AbstractLayer):
    def __init__(self, fshape, activation, filter_init, strides=1, pad=0):
        self.fshape = fshape
        self.strides = strides
        self.pad = pad
        self.filters = filter_init(self.fshape)
        self.activation = activation
        self.input = None
        self.a_slice = None

    def forward(self, inputs):
        self.inputs = inputs
        (m, n_H_prev, n_W_prev, n_C_prev) = inputs.shape
        (f, f, n_C_prev, n_C) = self.fshape
        n_H = int(np.floor((n_H_prev - f + 2 * self.pad) / self.strides) + 1)
        n_W = int(np.floor((n_W_prev - f + 2 * self.pad) / self.strides) + 1)
        fmap = np.zeros((m, n_H, n_W, n_C))
        inputs = zero_pad(inputs, self.pad)

        for h in xrange(n_H):
            for w in xrange(n_W):
                vert_start = h * self.strides
                vert_end = vert_start + f
                horiz_start = w * self.strides
                horiz_end = horiz_start + f

                fmap[:, h, w, :] = np.sum(
                    inputs[:, vert_start:vert_end, horiz_start:horiz_end, :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return self.activation.compute(fmap)

    def train_forward(self, inputs):
        self.inputs = inputs
        (m, n_H_prev, n_W_prev, n_C_prev) = inputs.shape
        (f, f, n_C_prev, n_C) = self.fshape
        n_H = int(np.floor((n_H_prev - f + 2 * self.pad) / self.strides) + 1)
        n_W = int(np.floor((n_W_prev - f + 2 * self.pad) / self.strides) + 1)
        fmap = np.zeros((m, n_H, n_W, n_C))
        inputs = zero_pad(inputs, self.pad)

        for h in xrange(n_H):
            for w in xrange(n_W):
                vert_start = h * self.strides
                vert_end = vert_start + f
                horiz_start = w * self.strides
                horiz_end = horiz_start + f

                fmap[:, h, w, :] = np.sum(inputs[:, vert_start:vert_end, horiz_start:horiz_end, :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return (fmap, self.activation.compute(fmap))

    def get_layer_error(self, z, backwarded_err):
        return backwarded_err * self.activation.deriv(z)

    def backward(self, layer_err):
        m, n_H, n_W, n_C = layer_err.shape
        m, n_H_prev, n_W_prev, n_C_prev = self.inputs.shape
        (f, f, n_C_prev, n_C) = self.fshape

        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        A_prev_pad = zero_pad(self.inputs, self.pad)
        dA_prev_pad = np.zeros(A_prev_pad.shape)

        for i in xrange(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in xrange(n_H):
                for w in xrange(n_W):
                    for c in xrange(n_C):
                        vert_start = h * self.strides
                        vert_end = vert_start + f
                        horiz_start = w * self.strides
                        horiz_end = horiz_start + f

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.filters[:, :, :, c] * layer_err[i, h, w, c]
            dA_prev[i, :, :, :] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]
        return dA_prev

    def get_grad(self, inputs, layer_err):

        m, n_H, n_W, n_C = layer_err.shape
        m, n_H_prev, n_W_prev, n_C_prev = self.inputs.shape
        (f, f, n_C_prev, n_C) = self.fshape

        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        A_prev_pad = zero_pad(self.inputs, self.pad)
        dA_prev_pad = np.zeros(A_prev_pad.shape)

        total_layer_err = np.sum(layer_err, axis=(0, 1, 2))
        filters_err = np.zeros(self.fshape)

        summed_inputs = np.sum(inputs, axis=0)
        for i in xrange(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in xrange(n_H):
                for w in xrange(n_W):
                    for c in xrange(n_C):
                        vert_start = h * self.strides
                        vert_end = vert_start + f
                        horiz_start = w * self.strides
                        horiz_end = horiz_start + f

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        filters_err[:,:,:, c] += a_slice * layer_err[i,h,w,c]
        return filters_err

    def update(self, grad):
        self.filters -= grad