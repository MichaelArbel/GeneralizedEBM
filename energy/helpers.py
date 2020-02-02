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
import model
import losses


from dde.conditional.r_data_loader import load_dataset

from dataloader import PrepareData

def get_data_loader(args):
	x_train, y_train, x_test, y_test = load_dataset(args.data_root,args.data_name)
	num_train = int(x_train.shape[0]*1.)
	valid_set = PrepareData(x_train[:num_train], y=y_train[:num_train])
	test_set = PrepareData(x_test, y=y_test)
	train_set = PrepareData(x_train[:num_train], y=y_train[:num_train])
	#valid_set = PrepareData(x_train[num_train:], y=y_train[num_train:])
	
	trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)
	testloader = torch.utils.data.DataLoader(test_set, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)
	validloader = torch.utils.data.DataLoader(valid_set, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)

	return trainloader,testloader,validloader


def get_loss(args):
	if args.criterion=='kale':
		return losses.kale
	else:
		raise NotImplementedError

# def get_reg(args, model):
# 	if args.regularizer=='L2':
# 		return L2_reg(model,args.reg_param)
# 	elif args.regularizer=='None':
# 		return None
# 	else:
# 		return None

def get_optimizer(args,params):
	if args.optimizer == 'Adam':
		optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, betas = (args.beta_1,args.beta_2))
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(params, lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
	else:
		raise NotImplementedError('optimizer {} not implemented'.format(args.optimizer))
	return optimizer

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

def get_latent(args,dim,device):
	dim = int(dim)
	if args.latent_noise=='gaussian':
		return torch.distributions.MultivariateNormal(torch.zeros(dim).to(device),torch.eye(dim).to(device))
	elif args.latent_noise=='uniform':
		return torch.distributions.Uniform(torch.zeros(dim).to(device),torch.ones(dim).to(device))

def get_discriminator(args,input_dims,device):
	return model.Discriminator(input_dims).to(device)

def get_generator(args,input_dims,device):
	if args.problem == 'conditional':
		net = model.GeneratorCond(input_dims).to(device)
	elif args.problem == 'unconditional':
		net = model.GeneratorUnCond(input_dims).to(device)
	return net


def get_net(args, net_type,device):
	if args.model=='resnet':
		_model = model_resnet
	elif args.model=='dcgan':
		_model = model_dcgan
	elif args.model=='sngan':
		_model = model
	else:
		raise NotImplementedError()	
	if net_type=='discriminator':
		net = _model.Discriminator().to(device)
	elif net_type =='generator':
		net = _model.Generator(args.Z_dim).to(device)
	return net
	




