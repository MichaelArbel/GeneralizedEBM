# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F

import torch
from spectral_normalization import SpectralNorm

channels = 3
leak = 0.1
w_g = 4

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

        out = self.linear_var*z
        out = self.linear_var.T*out
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


class GeneratorUnCond(nn.Module):
    def __init__(self, dims):
        super(GeneratorUnCond, self).__init__()
        self.z_dim = dims[0]

        self.linear_var =  nn.Parameter(torch.zeros([self.z_dim,self.z_dim]))
        self.bias = nn.Parameter(torch.zeros([self.z_dim]))
        self.lmbda = 1e-3
        self.Sigma= None
        self.dist = None

    def forward(self, z):
        out = self.linear_var*z
        out = self.linear_var.T*out
        out = out + self.lmbda*z + self.bias
        return out
    def log_density(self,x):
        if self.Sigma is None:
            self.Sigma = (self.linear_var.T*self.linear_var+self.lmbda* torch.eye([z_dim],device=self.linear_var.device))
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

        self.linear1 = nn.Linear(dim, 300)
        self.linear2 = nn.Linear(300, 400)
        self.linear3 = nn.Linear(400, 500)
        self.linear4 = nn.Linear(500, 1)

        #d_1,d_2,d_3, d_4, d_5 = 2000,3000,2000, 600,700
        d_1,d_2,d_3, d_4, d_5 = 1000,2000,1000, 600,700
        self.linear1 = nn.Linear(dim, d_1)
        self.linear2 = nn.Linear(d_1, d_2)
        self.linear3 = nn.Linear(d_2,d_3 )
        self.linear4 = nn.Linear(d_3, 1)
        # self.linear5 = nn.Linear(d_4, d_5)
        # self.linear6 = nn.Linear(d_5, 1)

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


