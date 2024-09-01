import numpy as np
# import tensorflow as tf
import torch
# from keras import backend as K


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = torch.sum(torch.tensor(~torch.isnan(x),dtype = torch.float32))
    return torch.tensor(torch.where(torch.equal(nelem, 0.), 1., nelem), dtype = x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.sum(x), nelem)


def mse_loss(y_true, y_pred):
    ret = torch.square(y_pred - y_true)

    return _reduce_mean(ret)


def poisson_loss(y_true, y_pred):
    y_pred = torch.tensor(y_pred, dtype = torch.float32)
    y_true = torch.tensor(y_true, dtype = torch.float32)
    nelem = _nelem(y_true)
    y_true = _nan2zero(y_true)
    ret = y_pred - y_true*torch.log(y_pred+1e-10) + torch.lgamma(y_true+1.0)

    return torch.divide(torch.sum(ret), nelem)



class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        y_true = torch.tensor(y_true, dtype = torch.float32)
        y_pred = torch.tensor(y_pred, dtype = torch.float32) * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

            # Clip theta
        theta = torch.minimum(self.theta,torch.tensor(1e6))

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))


        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.divide(torch.sum(final), nelem)
            else:
                final = torch.sum(final)


        return final

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps


            # reuse existing NB neg.log.lik.
            # mean is always False here, because everything is calculated
            # element-wise. we take the mean only in the end
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0-self.pi+eps)

        y_true = torch.tensor(y_true, dtype = torch.float32)
        y_pred = torch.tensor(y_pred, dtype = torch.float32) * scale_factor
        theta = torch.minimum(self.theta,torch.tensor(1e6))

        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        result = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda*torch.square(self.pi)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = torch.sum(result)

        result = _nan2inf(result)

        return result
