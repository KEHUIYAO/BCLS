from torch import nn
import torch


# Implementation of BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE by Yarin Gal
# the core idea is to set an approximating distribution modelling each kernel-patch pair with a distinct random variable, and this distribution randomly sets kernels to zero for different patches, which results in the equivalent explanation of applying dropout for each element in the tensor y before pooling. So implementing the bayesian CNN is therefore as simple as using dropout after every convolution layer before pooling

class BayesianDropout(nn.Module):
    def __init__(self, dropout, x):
        "if x is not none, generate dropout mask using x's shape"
        super().__init__()
        self.dropout = dropout

        self.m = x.new_empty(x.size()).bernoulli_(1 - dropout)


    def forward(self, x):
        "apply the dropout mask to x"
        x = x.masked_fill(self.m == 0, 0)
        return x


