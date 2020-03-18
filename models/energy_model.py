from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
channels = 3
leak = 0.1
w_g = 4

import models.made_model as made
from torch.nn.utils import spectral_norm as sn_official
spectral_norm = sn_official
class MADEGenerator(nn.Module):
    def __init__(self, dims):
        super(MADEGenerator, self).__init__()
        self.num_inputs = dims[0]
        self.num_hidden = 1000

        self.made = made.MADE(self.num_inputs,self.num_hidden,act='lrelu')

    def forward(self,z):
        u,a= self.made(z,mode='inverse')
        return u
    def log_density(self,x):
        u,a = self.made(x)
        return a +(-0.5*u.pow(2)- 0.5 * np.log(2 * np.pi)).sum(-1,keepdim=True)

class MAFGenerator(nn.Module):
    def __init__(self,dims):
        super(MAFGenerator, self).__init__()
        self.num_inputs = dims[0]

        self.made = made.FlowSequential(self.num_inputs)

    def forward(self,z):
        u,a= self.made(z,mode='inverse')
        return u
    def log_density(self,x):
        u,a = self.made(x)
        return a +(-0.5*u.pow(2)- 0.5 * np.log(2 * np.pi)).sum(-1,keepdim=True)


class GaussianGenerator(nn.Module):
    def __init__(self, dims):
        super(GaussianGenerator, self).__init__()
        self.z_dim = dims[0]

        self.linear_var =  nn.Parameter(torch.eye(int(self.z_dim)))
        self.bias = nn.Parameter(torch.zeros([self.z_dim]))
        self.lmbda = 1e-3
        self.Sigma= None
        self.dist = None

    def forward(self, z):
        out = z@self.linear_var.T
        out = out@self.linear_var
        out = out + self.lmbda*z + self.bias
        return out
    def log_density(self,x):
        if self.Sigma is None:
            self.Sigma = (self.linear_var.T*self.linear_var+self.lmbda* torch.eye(int(self.z_dim),device=self.linear_var.device))
            self.Sigma = self.Sigma*self.Sigma
            loc = torch.zeros([self.z_dim], device=self.linear_var.device)
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=self.Sigma)
        location = x - self.bias
        value = self.dist.log_prob(location)
        return value


class Discriminator(nn.Module):
    def __init__(self,dim,device):
        super(Discriminator, self).__init__()
        kernel_size = 3
        d_1,d_2,d_3, d_4, d_5,d_6 = 1000,2000,1000,300,200,100
        d_0 = int(dim)

        mask_1 = made.get_mask(d_0, d_1, d_0, mask_type='input')
        mask_2 = made.get_mask(d_1, d_2, d_0)
        mask_3 = made.get_mask(d_2, d_3, d_0, mask_type='output')


        self.linear1 = MaskedLinear(d_0, d_1, mask_1,device)
        self.linear2 = MaskedLinear(d_1, d_2, mask_2,device)
        self.linear3 = MaskedLinear(d_2, d_3, mask_3,device)
        #self.linear4 = spectral_norm(nn.Linear(d_3, 1))
        self.linear4 = nn.Linear(d_3, 1)
        self.max = 10


    def forward(self, x):
        m = x

        m = nn.LeakyReLU(leak)(self.linear1(m))
        m = nn.LeakyReLU(leak)(self.linear2(m))
        m = nn.LeakyReLU(leak)(self.linear3(m))

        m = self.linear4(m)
        m = nn.ReLU()(m+self.max)-self.max
        return m



class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask,device):
        super(MaskedLinear, self).__init__()

        #self.linear = spectral_norm(nn.Linear(in_features, out_features))
        self.linear = nn.Linear(in_features, out_features).to(device)
        self.register_buffer('mask', mask)
    def forward(self,x):
        out = F.linear(x, self.linear.weight * self.mask,
                          self.linear.bias)
        return out

