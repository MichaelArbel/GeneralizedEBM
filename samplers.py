import torch
import numpy as np
import time
import math
from torch.autograd import Variable
from torch.autograd.variable import Variable
from torch import nn

import torch.nn.functional as F
 
class Latent_potential(nn.Module):

    def __init__(self, generator,discriminator,latent_prior):
        #super(Latent_potential).__init__()
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_prior = latent_prior
    def forward(self,Z):
        out = self.generator(Z)
        out =  self.discriminator(out)
        #out = -self.latent_prior.log_prob(Z) - out 
        #out = 0.5 * torch.norm(Z, dim=1) ** 2 + out
        out = 0.5 * torch.norm(Z, dim=1) ** 2 + out

        return out

class Grad_potential(nn.Module):
    def __init__(self, potential):
        super().__init__()
        self.potential = potential
    def forward(self,X):
        X.requires_grad_()
        out = self.potential(X).sum()
        out.backward()
        return X.grad


class HMCsampler(object):
    def __init__(self, potential, momentum, sample_chain = False, T=100, num_steps_min=10, num_steps_max=20, gamma=1e-2,kappa = 4e-2):          
        
        self.momentum = momentum
        self.potential = potential
        
        self.num_steps_min = num_steps_min
        self.num_steps_max = num_steps_max
        self.kappa = kappa
        self.gamma = gamma
        self.sample_chain = sample_chain

        self.grad_potential = Grad_potential(self.potential)
        self.T = T
        self.dUdX = None
        #self.grad_momentum = Grad_potential(self.momentum.log_prob)
        #self.sampler_momentum = momentum_sampler 
        
    def sample(self,prior_z,sample_chain=False,T=None,thinning=10):
        if T is None:
            T = self.T
        sampler = torch.distributions.Normal(torch.zeros_like(prior_z), 1.)
        
        #self.momentum.eval()
        self.potential.eval()
        t_extract_list = []
        Z_extract_list = []
        avg_acceptence_list  = []

        #num_steps = np.random.randint(self.num_steps_min, self.num_steps_max + 1)
        num_steps =1
        U  =  torch.zeros([prior_z.shape[0]]).to(prior_z.device)

        Z_t = prior_z.clone().detach()
        V_t = torch.zeros_like(Z_t)
        #V_t = self.momentum.sample([Z_t.shape[0]])
        t_extract_list.append(0)
        Z_extract_list.append(Z_t)
        avg_acceptence_list.append(1.)

        Z_0 = prior_z.clone().detach()
        V_0 = torch.zeros_like(Z_t)
        gamma = self.gamma
        for t in range(1, T+1):
            # reset computation graph
            #V_t = self.momentum.sample([Z_t.shape[0]])
            #U = U.uniform_(0,1)
            #Z_new,V_new = self.leapfrog(Z_t,V_t,self.grad_potential,sampler,T=num_steps,gamma=self.gamma,kappa=self.kappa)
            #V_t = torch.zeros_like(Z_t)
            #Z_t,V_t = self.leapfrog(Z_t,V_t,self.grad_potential,sampler,T=num_steps,gamma=self.gamma,kappa=self.kappa)
            Z_t,V_t = self.leapfrog(Z_t,V_t,self.grad_potential,sampler,T=num_steps,gamma=gamma,kappa=self.kappa)
            if t>0 and t%200==0:
                gamma *=0.1
                print('decreasing lr for sampling')            
            #Z_tmp,V_tmp,acc_prob = self.hasing_metropolis(Z_new, V_new, Z_t, V_t, self.potential,self.momentum.log_prob, U)
            #Z_t,V_t,acc_prob = self.hasing_metropolis_2(Z_new, V_new, Z_t,V_t, Z_0, V_0, self.potential,self.momentum.log_prob, U)
            #Z_t = Z_new
            #V_t = V_new
            #Z_t,acc_prob = self.hasing_metropolis(Z_new, V_new, Z_t, V_t, self.potential,self.momentum.log_prob, U)
            #print( 'avg_acceptence: ' +  str(acc_prob.mean().item() ) + ', iteration: '+ str(t))
            # only if extracting the samples so we have a sequence of samples
            if sample_chain and thinning != 0 and t % thinning == 0:
                t_extract_list.append(t)
                Z_extract_list.append(Z_t.clone().detach().cpu())
                #avg_acceptence_list.append(acc_prob.mean().item())
                avg_acceptence_list.append(1.)
            #print('iteration: '+ str(t))
        if not sample_chain:
            return Z_t.clone().detach()
        return t_extract_list, Z_extract_list,avg_acceptence_list
    # def sample(self,prior_z,sample_chain=False,T=None,thinning=10):
    #     if T is None:
    #         T = self.T
    #     sampler = torch.distributions.Normal(torch.zeros_like(prior_z), 1.)
        
    #     #self.momentum.eval()
    #     self.potential.eval()
    #     t_extract_list = []
    #     Z_extract_list = []
    #     accept_proba_list = []


    #     num_steps = np.random.randint(self.num_steps_min, self.num_steps_max + 1)
    #     num_steps = 1
    #     U  =  torch.zeros([prior_z.shape[0]]).to(prior_z.device)

    #     Z_t = prior_z.clone().detach()
    #     t_extract_list.append(0)
    #     Z_extract_list.append(Z_t)
        
    #     for t in range(T):
    #         # reset computation graph
    #         V_t = self.momentum.sample([Z_t.shape[0]])
    #         U = U.uniform_(0,1)
    #         #Z_new,V_new = self.leapfrog(Z_t,V_t,self.grad_potential,sampler,T=num_steps,gamma=self.gamma,kappa=self.kappa)
    #         Z_new,V_new = self.leapfrog(Z_t,V_t,self.grad_potential,sampler,T=num_steps,gamma=self.gamma,kappa=self.kappa)
    #         V_new = -V_new
    #         Z_t,V_t,acc_prob = self.hasing_metropolis(Z_new, V_new, Z_t, V_t, self.potential,self.momentum.log_prob, U)
    #         # only if extracting the samples so we have a sequence of samples
    #         if sample_chain and thinning != 0 and t % thinning == 0:
    #             t_extract_list.append(t)
    #             Z_extract_list.append(Z_t.clone().detach().cpu())
    #             accept_proba_list.append(acc_prob.clone().detach().cpu())
    #         #print('iteration: '+ str(t))
    #     if not sample_chain:
    #         return Z_t.clone().detach()
    #     return t_extract_list, Z_extract_list


    def leapfrog(self,x,v,grad_x,sampler,T=100,gamma=1e-2,kappa=4e-2):
        x_t = x.clone().detach()
        v_t = v.clone().detach()

        C = np.exp(-kappa * gamma)
        D = np.sqrt(1 - np.exp(-2 * kappa * gamma))
        for t in range(T):
            # reset computation graph
            x_t.detach_()
            v_t.detach_()
            x_half = x_t + gamma / 2 * v_t
            # calculate potentials and derivatives
            dUdX = grad_x(x_half)
            # update values
            v_half = v_t - gamma / 2 * dUdX
            v_tilde = C * v_half + D * sampler.sample()
            #v_tilde  = v_half
            v_t = v_tilde - gamma / 2 * dUdX
            x_t = x_half + gamma / 2 * v_t

        return x_t, v_t

    # def leapfrog(self,x,v,grad_x,sampler,T=100,gamma=1e-2,kappa=4e-2):
    #     x_t = x.clone().detach()
    #     v_t = v.clone().detach()
    #     for t in range(T):
    #         # reset computation graph
    #         x_t.detach_()
    #         v_t.detach_()
            
    #         if self.dUdX is None:
    #             x_tmp = 1.*x_t
    #             self.dUdX = grad_x(x_tmp).clone()
    #         # update values
    #         v_half = v_t - gamma / 2 * self.dUdX
    #         v_half.detach_()
    #         x_out = x_t +   gamma *v_half
    #         self.dUdX = 1.*grad_x(x_out).clone()
    #         x_t = x_out.clone()
    #         v_t    = v_half - gamma / 2 * self.dUdX
    #     return x_t, v_t


    def hasing_metropolis(self,Z_new, V_new, Z_0, V_0, potential, momentum,U):
        momentum_0 = -momentum(V_0)
        potential_0 = potential(Z_0)
        potential_new = potential(Z_new)
        momentum_new = -momentum(V_new)

        H0 = potential_0 + momentum_0
        H  = potential_new + momentum_new 

        difference = -H + H0
        acc_prob = torch.exp(-F.relu(-difference))
        accepted = U < acc_prob
        Z_out = 1.*Z_0
        V_out = 1.*V_0
        Z_out[accepted] = 1.* Z_new[accepted]
        V_out[accepted] = 1.*V_new[accepted]
        return Z_out, V_out, acc_prob

    def hasing_metropolis_2(self,Z_new, V_new, Z_t,V_t, Z_0, V_0, potential, momentum,U):
        momentum_0 = -momentum(V_0)
        potential_0 = potential(Z_0)
        potential_new = potential(Z_new)
        momentum_new = -momentum(V_new)

        H0 = potential_0 + momentum_0
        H  = potential_new + momentum_new 

        difference = -H + H0
        acc_prob = torch.exp(-F.relu(-difference))
        accepted = U < acc_prob
        Z_out = 1.*Z_t
        V_out = 1.*V_t
        Z_out[accepted] = 1.* Z_new[accepted]
        V_out[accepted] = 1.*V_new[accepted]
        return Z_out, V_out, acc_prob

class LangevinSampler(object):
    def __init__(self, potential, momentum, sample_chain = False, T=100, num_steps_min=10, num_steps_max=20, gamma=1e-2,kappa = 4e-2):          
        
        self.momentum = momentum
        self.potential = potential
        
        self.num_steps_min = num_steps_min
        self.num_steps_max = num_steps_max
        self.kappa = kappa
        self.gamma = gamma
        self.sample_chain = sample_chain

        self.grad_potential = Grad_potential(self.potential)
        self.T = T
        #self.grad_momentum = Grad_potential(self.momentum.log_prob)
        #self.sampler_momentum = momentum_sampler 
        
    def sample(self,prior_z,sample_chain=False,T=None,thinning=10):
        if T is None:
            T = self.T
        sampler = torch.distributions.Normal(torch.zeros_like(prior_z), 1.)
        
        #self.momentum.eval()
        self.potential.eval()
        t_extract_list = []
        Z_extract_list = []
        accept_list = []
        num_steps = np.random.randint(self.num_steps_min, self.num_steps_max + 1)

        Z_t = prior_z.clone().detach()
        gamma = self.gamma
        for t in range( T):
            # reset computation graph
            Z_t = self.euler(Z_t,self.grad_potential,sampler,gamma=gamma)
            #Z_t,acc_prob = hasing_metropolis(Z_new, V_new, Z_t, V_t, self.potential,self.momentum.log_prob, U)
            # only if extracting the samples so we have a sequence of samples
            if t>0 and t%200==0:
                gamma *=0.1
                print('decreasing lr for sampling')
            if sample_chain and thinning != 0 and t % thinning == 0:
                t_extract_list.append(t)
                Z_extract_list.append(Z_t.clone().detach().cpu())
                accept_list.append(1.)
            #print('iteration: '+ str(t))
        if not sample_chain:
            return Z_t.clone().detach()
        return t_extract_list, Z_extract_list, accept_list


    def euler(self,x,grad_x,sampler,gamma=1e-2):
        x_t = x.clone().detach()
        D = np.sqrt(gamma)
        x_t = x_t - gamma / 2 * grad_x(x_t) + D * sampler.sample()
        return x_t








