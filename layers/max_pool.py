import numpy as np
from abstract_layer import AbstractLayer

class MaxPool(AbstractLayer):
    def __init__(self, f=1, strides=1, channels=8):
        self.f = f
        self.strides = strides
        self.channels = channels
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        (m, n_H_prev, n_W_prev, n_C_prev) = inputs.shape
        n_H = int(1 + (n_H_prev - self.f) / self.strides)
        n_W = int(1 + (n_W_prev - self.f) / self.strides)
        n_C = n_C_prev
        out = np.zeros((m, n_H, n_W, n_C))

        for h in xrange(n_H):
            for w in xrange(n_W):
                vert_start = h * self.strides
                vert_end = vert_start + self.f
                horiz_start = w * self.strides
                horiz_end = horiz_start + self.f

                out[:, h, w, :] = np.max(
                    inputs[:, vert_start:vert_end, horiz_start:horiz_end, :, np.newaxis],  axis=(1, 2, 3))
        return out

    def train_forward(self, inputs):
        self.inputs = inputs
        (m, n_H_prev, n_W_prev, n_C_prev) = inputs.shape
        n_H = int(1 + (n_H_prev - self.f) / self.strides)
        n_W = int(1 + (n_W_prev - self.f) / self.strides)
        n_C = n_C_prev
        out = np.zeros((m, n_H, n_W, n_C))

        for h in xrange(n_H):
            for w in xrange(n_W):
                vert_start = h * self.strides
                vert_end = vert_start + self.f
                horiz_start = w * self.strides
                horiz_end = horiz_start + self.f

                out[:, h, w, :] = np.max(
                    inputs[:, vert_start:vert_end, horiz_start:horiz_end, :, np.newaxis], axis=(1, 2, 3))
        return (out,out)

    def get_layer_error(self, z, backwarded_err):
        return backwarded_err

    def backward(self, layer_err):
        def create_mask_from_window(x): return (x == np.max(x)).astype(int)

        n_H_prev = self.f*layer_err.shape[1]
        n_W_prev = self.f*layer_err.shape[2]
        n_C_prev = self.channels
        m, n_H, n_W, n_C = layer_err.shape
        backwarded_fmap = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

        for i in range(m):
            for h in xrange(n_H):
                for w in xrange(n_W):
                    for c in xrange(n_C):
                        vert_start = h * self.strides
                        vert_end = vert_start + self.f
                        horiz_start = w * self.strides
                        horiz_end = horiz_start + self.f

                        input_slice = self.inputs[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(input_slice)
                        backwarded_fmap[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*layer_err[i,h,w,c]
        return backwarded_fmap

    def get_grad(self, inputs, layer_err):
        return 0

    def update(self, grad):
        pass