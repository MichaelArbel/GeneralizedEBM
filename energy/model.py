# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from spectral_normalization import SpectralNorm
from torch.autograd import Variable
channels = 3
leak = 0.1
w_g = 4

import made_model as made

class GeneratorCond(nn.Module):
    def __init__(self, dims):
        super(GeneratorCond, self).__init__()
        self.z_dim = dims[0]
        self.x_dim = dims[1]

        self.linear_var =  nn.Parameter(torch.eye(int(self.z_dim)))
        self.linear_x = nn.Linear(self.x_dim, self.z_dim)
        self.linear_x.weight.data = torch.zeros([int(self.z_dim),int(self.x_dim)]) 
        self.linear_x.bias.data = torch.zeros([int(self.z_dim)]) 
        self.lmbda = 1e-3
        self.Sigma = None
        self.dist  = None
    def forward(self, z,x,tile=0):

        out = z@self.linear_var.T
        out = out@self.linear_var
        out = out + self.lmbda*z + self.linear_x(x)
        return out
    def log_density(self,y,x):
        if self.Sigma is None:
            self.Sigma = (self.linear_var.T*self.linear_var+self.lmbda* torch.eye(int(self.z_dim),device=self.linear_var.device))
            self.Sigma = self.Sigma*self.Sigma
            loc = torch.zeros([self.z_dim], device=self.linear_var.device)
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=self.Sigma)
        location = y - self.linear_x(x)
        value = self.dist.log_prob(location)
        return value

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


class GeneratorUnCond(nn.Module):
    def __init__(self, dims):
        super(GeneratorUnCond, self).__init__()
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
    def __init__(self,dim):
        super(Discriminator, self).__init__()

        # self.linear1 = SpectralNorm(nn.Linear(dim, 64))
        # self.linear2 = SpectralNorm(nn.Linear(64, 128))
        # self.linear3 = SpectralNorm(nn.Linear(128, 256))
        # self.linear4 = SpectralNorm(nn.Linear(256, 1))

        #self.linear1 = nn.Linear(dim, 300)
        #self.linear2 = nn.Linear(300, 400)
        #self.linear3 = nn.Linear(400, 500)
        #self.linear4 = nn.Linear(500, 1)

        #d_1,d_2,d_3, d_4, d_5 = 2000,3000,2000, 600,700
        d_1,d_2,d_3, d_4, d_5 = 1000,2000,1000, 2000,1000
        
        self.linear1 = nn.Linear(dim, d_1)
        self.linear2 = nn.Linear(d_1, d_2)
        self.linear3 = nn.Linear(d_2, d_3)
        self.linear4 = nn.Linear(d_3, 1)
        #self.linear5 = nn.Linear(d_4, d_5)
        #self.linear6 = nn.Linear(d_5, 1)

    def forward(self, x):
        m = x
        #m = torch.exp(self.linear1(m))
        #m = self.linear2(m)
        m = nn.LeakyReLU(leak)(self.linear1(m))
        m = nn.LeakyReLU(leak)(self.linear2(m))
        m = nn.LeakyReLU(leak)(self.linear3(m))
        #m = nn.LeakyReLU(leak)(self.linear4(m))
        #m = nn.LeakyReLU(leak)(self.linear5(m))
        m = self.linear4(m)

        return m

class Discriminator(nn.Module):
    def __init__(self,dim):
        super(Discriminator, self).__init__()

        # self.linear1 = SpectralNorm(nn.Linear(dim, 64))
        # self.linear2 = SpectralNorm(nn.Linear(64, 128))
        # self.linear3 = SpectralNorm(nn.Linear(128, 256))
        # self.linear4 = SpectralNorm(nn.Linear(256, 1))

        #self.linear1 = nn.Linear(dim, 300)
        #self.linear2 = nn.Linear(300, 400)
        #self.linear3 = nn.Linear(400, 500)
        #self.linear4 = nn.Linear(500, 1)

        #d_1,d_2,d_3, d_4, d_5 = 2000,3000,2000, 600,700
        d_1,d_2,d_3, d_4, d_5 = 100,200,100, 500,100
        
        self.linear1 = nn.Linear(dim, d_1, bias=False)
        self.linear2 = nn.Linear(d_1, d_2, bias=False)
        self.linear3 = nn.Linear(d_2, d_3, bias=False)
        self.linear4 = nn.Linear(d_3, d_4, bias=False)
        self.linear_skip = nn.Linear(dim, d_3, bias=False)
        self.linear5 = nn.Linear(d_4, d_5, bias=False)
        self.linear6 = nn.Linear(d_5, 1, bias=False)

    def forward(self, x):
        m = x
        #m = torch.exp(self.linear1(m))
        #m = self.linear2(m)
        m = nn.LeakyReLU(leak)(self.linear1(m))
        m = nn.LeakyReLU(leak)(self.linear2(m))
        m = nn.LeakyReLU(leak)(self.linear3(m))

        m = nn.LeakyReLU(leak)(self.linear4(m))
        m = nn.ELeakyReLULU(leak)(self.linear5(m))
        m = self.linear6(m)

        return m


class Discriminator(nn.Module):
    def __init__(self,dim):
        super(Discriminator, self).__init__()
        kernel_size = 3
        # self.linear1 = SpectralNorm(nn.Linear(dim, 64))
        # self.linear2 = SpectralNorm(nn.Linear(64, 128))
        # self.linear3 = SpectralNorm(nn.Linear(128, 256))
        # self.linear4 = SpectralNorm(nn.Linear(256, 1))
        d_1,d_2,d_3, d_4, d_5 = 1000,2000,1000, 500,100
        
        self.linear1 = MaskedLinear(int(dim), d_1, kernel_size)
        self.linear2 = MaskedLinear(d_1, d_2, kernel_size*20)
        self.linear3 = nn.Linear(d_2, d_3)
        self.linear4 = nn.Linear(d_3, 1)
        #self.linear_skip = nn.Linear(dim, d_3, kernel_size)
        #self.linear5 = MaskedLinear(d_4, d_5, kernel_size)
        #self.linear6 = nn.Linear(d_5, 1)


    def forward(self, x):
        m = x
        #m = torch.exp(self.linear1(m))
        #m = self.linear2(m)



        m = nn.LeakyReLU(leak)(self.linear1(m))
        m = nn.LeakyReLU(leak)(self.linear2(m))
        m = nn.LeakyReLU(leak)(self.linear3(m))

        #m = nn.LeakyReLU(leak)(self.linear4(m))
        #m = nn.LeakyReLU(leak)(self.linear5(m))
        m = self.linear4(m)

        return m

    def make_masks(self,x,w):
        torch.zeros()

class Discriminator(nn.Module):
    def __init__(self,dim):
        super(Discriminator, self).__init__()
        kernel_size = 3
        # self.linear1 = SpectralNorm(nn.Linear(dim, 64))
        # self.linear2 = SpectralNorm(nn.Linear(64, 128))
        # self.linear3 = SpectralNorm(nn.Linear(128, 256))
        # self.linear4 = SpectralNorm(nn.Linear(256, 1))
        d_1,d_2,d_3, d_4, d_5,d_6 = 1000,2000,1000,300,200,100
        d_0 = int(dim)

        mask_1 = made.get_mask(d_0, d_1, d_0, mask_type='input')
        mask_2 = made.get_mask(d_1, d_2, d_0)
        mask_3 = made.get_mask(d_2, d_3, d_0, mask_type='output')
        #mask_3 = made.get_mask(d_2, d_3, d_0, mask_type='output')
        #mask_3 = made.get_mask(d_2, d_3, d_0, mask_type='output')

        self.linear1 = made.MaskedLinear(d_0, d_1, mask_1)
        self.linear2 = made.MaskedLinear(d_1, d_2, mask_2)
        self.linear3 = made.MaskedLinear(d_2, d_3, mask_3)
        #self.linear3 = made.MaskedLinear(d_2, d_3, output_mask)

        # self.linear1 = nn.Linear(d_0, d_1)
        # self.linear2 = nn.Linear(d_1, d_2)
        # self.linear3 = nn.Linear(d_2, d_3)
        
        
        #self.linear2 = MaskedLinear(d_1, d_2, kernel_size*20)
        self.linear4 = nn.Linear(d_3, 1)
        #self.linear_skip = nn.Linear(dim, d_3, kernel_size)
        #self.linear5 = MaskedLinear(d_4, d_5, kernel_size)
        #self.linear6 = nn.Linear(d_5, 1)


    def forward(self, x):
        m = x
        #m = torch.exp(self.linear1(m))
        #m = self.linear2(m)

        m = nn.LeakyReLU(leak)(self.linear1(m))
        m = nn.LeakyReLU(leak)(self.linear2(m))
        m = nn.LeakyReLU(leak)(self.linear3(m))

        #m = nn.LeakyReLU(leak)(self.linear4(m))
        #m = nn.LeakyReLU(leak)(self.linear5(m))
        m = self.linear4(m)

        return m

    def make_masks(self,x,w):
        torch.zeros()



class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super(MaskedLinear, self).__init__()
        # sample random masks
        mask_int = torch.multinomial(torch.ones([out_features,in_features]),kernel_size, replacement=False)
        mask = torch.nn.functional.one_hot(mask_int,in_features)
        mask = mask.sum([1]).type(torch.float32)

        self.linear = nn.Linear(in_features, out_features)
        #self.linear.weights *= mask  # to zero it out first
        self.mask = nn.Parameter(mask, requires_grad=False)
        #self.handle = self.register_backward_hook(zero_grad)  # to make sure gradients won't propagate
    def forward(self,x):
        masked_weight = self.linear.weight*self.mask
        out = x@masked_weight.T + self.linear.bias
        return out

