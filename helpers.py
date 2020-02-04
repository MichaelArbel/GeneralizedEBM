import argparse
import copy
import hashlib
import json
import numpy as np
import os
import time
#import thop
import ast
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.generator import Generator
from models.discriminator import Discriminator
import losses

import pdb
import time


def get_data_loader(args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.dataset == 'cifar10':
        spatial_size = 32

        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)
        n_classes=10
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.b_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'imagenet32' or args.dataset=='imagenet64':
        normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                                         std=[0.5,0.5,0.5])
        from imagenet import Imagenet32

        spatial_size = 32
        if args.dataset=='imagenet64':
            spatial_size = 64
        n_classes = 1000

        trainset = Imagenet32(args.path_train, transform=transforms.Compose(transforms_train), sz=spatial_size)
        valset = Imagenet32(args.path_test, transform=transforms.Compose(transforms_test), sz=spatial_size)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(valset,batch_size=args.b_size, shuffle=False,num_workers=args.num_workers)
        n_classes = 1000
    else:
        raise NotImplementedError
    return trainloader,testloader


def get_fid_stats(dataset):
    if dataset=='cifar10_train':
        path = 'metrics/res/stats_pytorch/fid_stats_cifar10_train.npz'
    elif dataset=='cifar10_valid':
        path = 'metrics/res/stats_pytorch/fid_stats_cifar10_valid.npz'
    elif dataset=='imagenet_train':
        path = 'metrics/res/stats_pytorch/fid_stats_imagenet_train.npz'
    elif dataset=='imagenet_train':
        path = 'metrics/res/stats_pytorch/fid_stats_imagenet_valid.npz'
    elif dataset=='celeba':
        path = 'metrics/res/stats_pytorch/fid_stats_celeba.npz'

    f = np.load(path)
    mu1, sigma1 = f['mu'][:], f['sigma'][:]
    f.close()       
    return mu1, sigma1

def get_fid_stats_pytorch(dataset):
    if dataset=='cifar10':
        path = 'metrics/res/stats_pytorch/fid_stats_cifar10_train.npz'
    elif dataset=='imagenet_train':
        path = 'metrics/res/stats_pytorch/fid_stats_imagenet_train.npz'
    elif dataset=='imagenet_valid':
        path = 'metrics/res/stats_pytorch/fid_stats_imagenet_valid.npz'
    elif dataset=='celeba':
        path = 'metrics/res/stats_pytorch/fid_stats_celeba.npz'

    return path



def get_loss(args):
    if args.criterion=='hinge':
        return losses.hinge
    elif args.criterion=='wasserstein':
        return losses.wasserstein
    elif args.criterion=='logistic':
        return losses.logistic
    elif args.criterion=='kale':
        return losses.kale
    elif args.criterion=='kale_np':
        return losses.kale

# def get_reg(args, model):
#   if args.regularizer=='L2':
#       return L2_reg(model,args.reg_param)
#   elif args.regularizer=='None':
#       return None
#   else:
#       return None

# choose the optimizer
def get_optimizer(args,params):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, betas = (args.beta_1,args.beta_2))
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('optimizer {} not implemented'.format(args.optimizer))
    return optimizer

# schedule the learning
def get_scheduler(args,optimizer):
    if args.scheduler=='MultiStepLR':
        if args.milestone is None:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.total_epochs*0.5), int(args.total_epochs*0.75)], gamma=args.lr_decay)
        else:
            milestone = [int(_) for _ in args.milestone.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=args.lr_decay)
        return lr_scheduler
    elif args.scheduler=='ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

# return some sort of latent noise
def get_latent(args,device):
    if args.latent_noise=='gaussian':
        return torch.distributions.Normal(torch.zeros(args.Z_dim).to(device),1)
    elif args.latent_noise=='uniform':
        return torch.distributions.Uniform(torch.zeros(args.Z_dim).to(device),torch.ones(args.Z_dim).to(device))


# get posterior samples using MLE information learned by the GAN
def get_latent_samples(prior_z, s_type, g=None, h=None, sampler=None, gamma=1e-2, kappa=4e-2, T=10):
    if s_type == 'none':
        # don't sample from the posterior
        return prior_z
    else:
        # we'll be doing some kind of monte carlo sampling using the learned MLE
        
        # don't use batchnorm. maybe get rid of this later if ths function used correctly
        h.eval()
        g.eval()

        def U_potential(z, h, g):
            return 1/2 * torch.norm(z, dim=1) ** 2 + h(g(z))

        # Zs are the samples
        Z_t = prior_z.clone().detach()

        if s_type == 'lmc':

            # langevin monte carlo
            V_t = torch.zeros_like(Z_t)
            C = np.exp(-kappa * gamma)
            D = np.sqrt(1 - np.exp(-2 * kappa * gamma))
            for t in range(T):
                
                # reset computation graph
                Z_t.detach_()
                V_t.detach_()
                Z_half = Z_t + gamma / 2 * V_t
                Z_half.requires_grad_()
                # calculate potentials and derivatives
                U = U_potential(Z_half, h, g).sum()
                U.backward()
                dUdZ = Z_half.grad
                # update values
                V_half = V_t - gamma / 2 * dUdZ
                V_tilde = C * V_half + D * sampler.sample()
                V_t = V_tilde - gamma / 2 * dUdZ
                Z_t = Z_half + gamma / 2 * V_t
                

        if s_type.startswith('mmc'):
            # michael monte carlo
            if s_type == 'mmc1':
                C = 0
            elif s_type == 'mmc2':
                C = np.sqrt(2 * gamma)
            for t in range(T):
                Z_t.requires_grad_()
                U = U_potential(Z_t, h, g).sum()
                U.backward()
                dUdZ = Z_t.grad.detach()
                Z_new = Z_t - gamma * dUdZ
                if s_type == 'mmc2':
                    Z_new += C * sampler.sample()
                Z_t = Z_t - gamma * dUdZ
                Z_t.detach_()

        h.train()
        g.train()
        Z_t.detach_()

        # we just want the prior, and the last sample
        return Z_t



# initialize neural net corresponding to type
def get_net(args, net_type, device):
    if net_type == 'discriminator':
        net = Discriminator(nn_type=args.d_model).to(device)
    elif net_type == 'generator':
        net = Generator(nz=args.Z_dim, nn_type=args.g_model).to(device)
    return net
    




