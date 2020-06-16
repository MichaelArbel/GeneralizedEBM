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


from models.mog_maf_model import MAFMOG, MAF, MADE

import models.mog_maf_model as mms

from torch.nn.utils import spectral_norm as sn_official
spectral_norm = sn_official

class CombinedDiscriminator(nn.Module):
    def __init__(self, discriminator, generator):
        super(CombinedDiscriminator, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def forward(self,x):
        out = self.discriminator(x)+self.generator.log_density(x)
        return out
    def log_density(self,x):
        try:
            out = self.discriminator.log_density(x)- self.generator.log_density(x)
            return out
        except:
            return torch.sum(torch.zeros_like(x),dim=1)
    def log_partition(self):
        try:

            return self.discriminator.log_partition()
        except:
            return torch.tensor(0.)

class Identity(nn.Module):
    def __init__(self,max_val):
        super(Identity, self).__init__()
        self.max = max_val
    def forward(self,x):
        return x
    def inverse(self,x):
        return x
    def log_grad(self,x):
        return torch.zeros_like(x)





class MADEGenerator(nn.Module):
    def __init__(self, dims, mode= 'discriminator'):
        super(MADEGenerator, self).__init__()
        self.num_inputs = dims[0]
        self.num_hidden = 100
        self.mode = mode
        self.made = made.MADE(self.num_inputs,self.num_hidden,act='lrelu')
        self.params = list(self.made.parameters())
    def forward(self,z):
        if self.mode=='generator':
            u,a= self.made(z,mode='inverse')
            return u
        elif self.mode=='discriminator':
            return -self.log_density(z) -self.log_partition() 
    def log_density(self,x):
        u,a = self.made(x)
        return a +(-0.5*u.pow(2)- 0.5 * np.log(2 * np.pi)).sum(-1,keepdim=True)

    def log_partition(self):
        abs_value = [torch.mean(torch.abs(p)) for p in self.params]
        abs_value = torch.stack(abs_value,dim=0).sum()
        return -abs_value


class NVP(nn.Module):
    def __init__(self,dims, device,num_blocks, mode = 'generator', with_bn=True):
        super(NVP, self).__init__()
        num_inputs = dims[0]
        num_hidden = 100
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()
        num_cond_inputs = None
        modules = []
        num_blocks = 5
        for _ in range(num_blocks):
            if with_bn:   
                modules += [
                    made.CouplingLayer(
                        num_inputs, num_hidden, mask, num_cond_inputs,
                        s_act='tanh', t_act='relu'),
                    made.BatchNormFlow(num_inputs, device=device).to(device)
                ]
            else:
                modules += [
                    made.CouplingLayer(
                        num_inputs, num_hidden, mask, num_cond_inputs,
                        s_act='tanh', t_act='relu')
                ]
            mask = 1 - mask
        self.model = made.FlowSequential(*modules)
        self.max = 10
        self.non_linearity = Identity(self.max)
        self.mode = mode
        self.params = list(self.model.parameters())
    def forward(self,z):
        if self.mode=='generator':
            z = self.non_linearity.inverse(z)
            u,a= self.model(z,mode='inverse')
            return u
        elif self.mode=='discriminator':
            return -self.log_density(z) -self.log_partition() 
    def log_density(self,x):
        u, log_jacob = self.model(x)
        log_probs = self.non_linearity.log_grad(u).sum(-1,keepdim=True) + log_jacob
        log_probs += (-0.5 * self.non_linearity(u).pow(2) - 0.5 * np.log(2 * np.pi)).sum(
            -1, keepdim=True)
        return log_probs

    def log_partition(self):
        abs_value = [torch.mean(torch.abs(p)) for p in self.params]
        abs_value = torch.stack(abs_value,dim=0).sum()
        return -abs_value

class FlowGenerator(nn.Module):
    def __init__(self,dims, device,num_blocks,flow_type, mode = 'generator', with_bn=True):
        super(FlowGenerator, self).__init__()
        num_inputs = dims[0]
        hidden_size = 100
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()
        num_cond_inputs = None
        modules = []
        num_blocks = 5
        self.n_components = 10
        n_hidden = 1
        if flow_type=='mogmaf':
            self.model = MAFMOG(num_blocks,self.n_components, num_inputs,hidden_size,n_hidden,batch_norm=with_bn)
        elif flow_type=='maf':
            self.model = MAF(num_blocks, num_inputs,hidden_size,n_hidden,batch_norm=with_bn)
        elif flow_type=='made':
            self.model = made.MADE(num_inputs,hidden_size, act= 'lrelu')
        self.max = 10
        self.non_linearity = Identity(self.max)
        self.mode = mode
        self.params = list(self.model.parameters())
    def forward(self,z):
        if self.mode=='generator':
            Z = z.unsqueeze(1).repeat(1,self.n_components,1)
            u,a= self.model.inverse(Z)
            return u
        elif self.mode=='discriminator':
            return -self.log_density(z) -self.log_partition() 
    def log_density(self,x):
        return self.model.log_prob(x)

    def log_partition(self):
        abs_value = [torch.mean(torch.abs(p)) for p in self.params]
        abs_value = torch.stack(abs_value,dim=0).sum()
        return -abs_value

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
            self.Sigma = self.Sigma@self.Sigma.T
            loc = torch.zeros([self.z_dim], device=self.linear_var.device)
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=self.Sigma)
        location = x - self.bias

        value = self.dist.log_prob(location)
        return value

class GaussianGenerator(nn.Module):
    def __init__(self, dims):
        super(GaussianGenerator, self).__init__()
        self.z_dim = dims[0]

        self.linear_var =  nn.Parameter(1.*torch.ones([self.z_dim]))
        self.bias = nn.Parameter(torch.zeros([self.z_dim]))
        self.lmbda = 1e-3
        #self.Sigma= None
        self.dist = None

    def forward(self, z):
        out = z*self.linear_var**2
        out = out + self.lmbda*z + self.bias
        return out
    def log_density(self,x):
        Sigma = self.linear_var**2 + self.lmbda
        Sigma = Sigma**2
        location = (x - self.bias)
        quad = torch.einsum('nd,nd,d->n',location,location,1./Sigma)
        quad = quad.unsqueeze(-1)
        value = (-0.5 * quad - 0.5 *torch.log(2.*np.pi*Sigma).sum()   )
        return value




# class NVPDiscriminator(nn.Module):
#     def __init__(self,dims, device,num_blocks):
#         num_inputs = dims[0]
#         num_hidden = 3
#         mask = torch.arange(0, self.num_inputs) % 2
#         mask = mask.to(device).float()
#         num_cond_inputs = None
#         for _ in range(args.num_blocks):
#             modules += [
#                 fnn.CouplingLayer(
#                     num_inputs, num_hidden, mask, num_cond_inputs,
#                     s_act='tanh', t_act='relu'),
#                 fnn.BatchNormFlow(num_inputs)
#             ]
#             mask = 1 - mask
#         self.model = made.FlowSequential(*modules)
#         self.params = list(self.model.parameters())
#     def forward(self,x):
#         abs_value = [torch.mean(torch.abs(p)) for p in self.params]
#         abs_value = torch.cat(abs_value,dim=0).mean()

#         return -self.log_density(x) + abs_value
#     def log_density(self,x):
#         return self.log_probs(x)


class Discriminator(nn.Module):
    def __init__(self,dim,device):
        super(Discriminator, self).__init__()
        kernel_size = 3
        d_out = 100
        d_in = int(dim)

        self.max = 10
        self.leak = 0.1
        n_hidden = 1
        n_blocks = 5

        modules = []
        modules += [self.make_masked_linear(d_in, d_out, n_hidden, device)]

        for i in range(n_blocks-1):
            modules += [ nn.LeakyReLU(self.leak) , self.make_masked_linear(d_in, d_out, n_hidden, device)]

        modules +=[ nn.LeakyReLU(self.leak) , nn.Linear(d_in, 1) ]

        self.net = nn.Sequential(*modules)


    def forward(self, x):
        out = self.net(x)
        out = nn.ReLU()(out+self.max)-self.max
        return out


    def make_masked_linear(self,d_in,d_out,n_hidden, device):
        masks,_ = mms.create_masks(d_in, d_out,  n_hidden, input_order = 'sequential')
        net = []
        net.append( mms.MaskedLinear(d_in, d_out, masks[0] ))
        for m in masks[1:-1]:
            net += [nn.LeakyReLU(self.leak)  , mms.MaskedLinear(d_out, d_out, m)] 
        net += [nn.LeakyReLU(self.leak)  , mms.MaskedLinear(d_out, d_in, masks[-1])] 

        net = nn.Sequential(*net)
        return net


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


class Discriminator(nn.Module):
    def __init__(self,dim,device):
        super(Discriminator, self).__init__()
        kernel_size = 3
        d_out = 100
        d_in = int(dim)

        self.max = 6
        self.leak = 0.1
        n_hidden = 1
        n_blocks = 5

        modules = []
        modules += [self.make_linear(d_in, d_out, n_hidden, device)]

        for i in range(n_blocks-1):
            modules += [ nn.LeakyReLU(self.leak) , self.make_linear(d_in, d_out, n_hidden, device)]

        modules +=[ nn.LeakyReLU(self.leak) , spectral_norm(nn.Linear(d_in, 1)) ]

        self.net = nn.Sequential(*modules)


    def forward(self, x):
        out = self.net(x)
        out = nn.ReLU()(out+self.max)-self.max
        return out


    def make_linear(self,d_in,d_out,n_hidden, device):
        #masks,_ = mms.create_masks(d_in, d_out,  n_hidden, input_order = 'sequential')
        net = []
        net.append( spectral_norm(nn.Linear(d_in, d_out )))
        for m in range(n_hidden):
            net += [nn.LeakyReLU(self.leak)  ,spectral_norm(nn.Linear(d_out, d_out))] 
        net += [nn.LeakyReLU(self.leak)  , spectral_norm(nn.Linear(d_out, d_in))] 

        net = nn.Sequential(*net)
        return net


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


class Discriminator(nn.Module):
    def __init__(self,dim,device):
        super(Discriminator, self).__init__()
        kernel_size = 3
        d_1,d_2,d_3= 100,200,100
        d_0 = int(dim)

        mask_1 = made.get_mask(d_0, d_1, d_0+1, mask_type='input')
        mask_2 = made.get_mask(d_1, d_2, d_0)
        #mask_2 = made.get_mask(d_2, d_3, d_0)
        mask_3 = made.get_mask(d_2, d_3, d_0, mask_type='output')


        self.linear1 = MaskedLinear(d_0, d_1, mask_1,device)
        self.linear2 = MaskedLinear(d_1, d_2, mask_2,device)
        self.linear3 = MaskedLinear(d_2, d_3, mask_3,device)
        #self.linear4 = MaskedLinear(d_4, d_4, mask_4,device)


        #self.linear1 = Linear(d_0, d_1, mask_1,device)
        #self.linear2 = Linear(d_1, d_2, mask_2,device)
        #self.linear3 = Linear(d_2, d_3, mask_3,device)


        #self.linear4 = spectral_norm(nn.Linear(d_3, 1))
        self.linear4 = spectral_norm(nn.Linear(d_3, 1))
        self.max = 10


    def forward(self, x):
        m = x

        m = nn.LeakyReLU(leak)(self.linear1(m))
        m = nn.LeakyReLU(leak)(self.linear2(m))
        m = nn.LeakyReLU(leak)(self.linear3(m))
        #m = nn.LeakyReLU(leak)(self.linear4(m))

        m = self.linear4(m)
        m = nn.ReLU()(m+self.max)-self.max
        return m



class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask,device):
        super(MaskedLinear, self).__init__()

        #self.linear = spectral_norm(nn.Linear(in_features, out_features))
        self.linear = spectral_norm(nn.Linear(in_features, out_features)).to(device)
        self.register_buffer('mask', mask)
    def forward(self,x):
        out = F.linear(x, self.linear.weight * self.mask,
                          self.linear.bias)
        return out


class Discriminator4(nn.Module):
    def __init__(self,dim,device):
        super(Discriminator4, self).__init__()
        kernel_size = 3
        d_1,d_2,d_3,d_4= 100,200,200,100
        d_0 = int(dim)

        mask_1 = made.get_mask(d_0, d_1, d_0+1, mask_type='input')
        mask_2 = made.get_mask(d_1, d_2, d_0)
        mask_3 = made.get_mask(d_2, d_3, d_0)
        mask_4 = made.get_mask(d_3, d_4, d_0, mask_type='output')


        self.linear1 = MaskedLinear(d_0, d_1, mask_1,device)
        self.linear2 = MaskedLinear(d_1, d_2, mask_2,device)
        self.linear3 = MaskedLinear(d_2, d_3, mask_3,device)
        self.linear4 = MaskedLinear(d_3, d_4, mask_4,device)



        self.linear5 = nn.Linear(d_4, 1)
        self.max = 10


    def forward(self, x):
        m = x

        m = nn.LeakyReLU(leak)(self.linear1(m))
        m = nn.LeakyReLU(leak)(self.linear2(m))
        m = nn.LeakyReLU(leak)(self.linear3(m))
        m = nn.LeakyReLU(leak)(self.linear4(m))

        m = self.linear5(m)
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


