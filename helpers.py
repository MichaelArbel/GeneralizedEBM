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
from models import energy_model

import compute as cp

import time

from dataloader import load_data, PrepareUCIData

# choose dataloaders for pytorch
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

    else:
        raise NotImplementedError
    return trainloader,testloader


def get_data_loader_energy(args):
    p = load_data(args.data_name)

    train_data = p.data
    test_data = p.test_data
    valid_data = p.valid_data
    log_volume =  np.sum(np.log(p.s))
    train_set = PrepareUCIData(train_data)
    test_set = PrepareUCIData(test_data)
    valid_set = PrepareUCIData(valid_data)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)

    return trainloader,testloader,validloader, log_volume




# choose loss type
def get_loss(args):
    if args.criterion=='hinge':
        return cp.hinge
    elif args.criterion=='wasserstein':
        return cp.wasserstein
    elif args.criterion=='logistic':
        return cp.logistic
    elif args.criterion=='kale':
        return cp.kale
    elif args.criterion=='kale-nlp':
        return cp.kale

# choose the optimizer
def get_optimizer(args, net_type, params):
    if net_type == 'discriminator':
        learn_rate = args.lr
    elif net_type == 'generator':
        learn_rate = args.lr_generator
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=learn_rate, weight_decay=args.weight_decay, betas = (args.beta_1,args.beta_2))
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=learn_rate, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
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
def get_normal(args, device, b_size):
    return torch.distributions.Normal(torch.zeros([b_size, args.Z_dim]).to(device), 1)

# return the distribution of the latent noise
def get_latent(args,dim,device):
    dim = int(dim)
    if args.latent_noise=='gaussian':
        return torch.distributions.MultivariateNormal(torch.zeros(dim).to(device),torch.eye(dim).to(device))
    elif args.latent_noise=='uniform':
        return torch.distributions.Uniform(torch.zeros(dim).to(device),torch.ones(dim).to(device))


# initialize neural net corresponding to type
def get_net(args, net_type, device):
    if net_type == 'discriminator':
        net = Discriminator(nn_type=args.d_model).to(device)
    elif net_type == 'generator':
        net = Generator(nz=args.Z_dim, nn_type=args.g_model).to(device)
    return net
    

# return a discriminator for the energy model
def get_discriminator(args,input_dims,device):
    return energy_model.Discriminator(input_dims,device).to(device)

# return the base for the energy model
def get_base_energy(args,input_dims,device):
    if args.generator == 'gaussian':
        net = energy_model.GaussianGenerator(input_dims).to(device)
    elif args.generator == 'made':
        net = energy_model.MADEGenerator(input_dims).to(device)
    return net


# choose device
def assign_device(device):
    if device >-1:
        device = 'cuda:'+str(device) if torch.cuda.is_available() and device>-1 else 'cpu'
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device


