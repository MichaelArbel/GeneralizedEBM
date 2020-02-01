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
import model, model_resnet, model_dcgan
import losses


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

def get_latent(args,device):
	if args.latent_noise=='gaussian':
		return torch.distributions.MultivariateNormal(torch.ones(args.Z_dim).to(device),torch.eye(args.Z_dim).to(device))
	elif args.latent_noise=='uniform':
		return torch.distributions.Uniform(torch.zeros(args.Z_dim).to(device),torch.ones(args.Z_dim).to(device))

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
	




