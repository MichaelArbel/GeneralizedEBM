import argparse
import copy
import hashlib
import json
import numpy as np
import os
import time

import ast
from torchvision.datasets import CIFAR10,ImageNet,DatasetFolder,LSUN
import torchvision.transforms as transforms

from utils.celebA import CelebA

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
from PIL import Image, ImageFilter
from utils.dataloader import load_data, PrepareUCIData
import samplers
import sys
import models.toy_models as tm
#from pytorch_pretrained_biggan import truncated_noise_sample


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
# choose dataloaders for pytorch
def get_image_loader(args,b_size,num_workers):
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        n_classes=10
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=num_workers)
    elif args.dataset == 'lsun':
        spatial_size = 32

        transform_lsun = transforms.Compose([
            transforms.Resize(spatial_size),
            transforms.CenterCrop(spatial_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        trainset = LSUN(
            root = args.data_path,
            classes = ['bedroom_train'],
            transform = transform_lsun)
        testset = LSUN(
            root = args.data_path,
            classes = ['bedroom_val'],
            transform = transform_lsun)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=True, num_workers=num_workers)

    elif args.dataset == 'imagenet32':

        from utils.imagenet import Imagenet32

        spatial_size = 32

        transforms_train = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        trainset = Imagenet32(args.imagenet_train_path, transform=transforms.Compose(transforms_train), sz=spatial_size)
        valset = Imagenet32(args.imagenet_test_path, transform=transforms.Compose(transforms_train), sz=spatial_size)
        n_classes=1000
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(valset, batch_size=b_size, shuffle=False, num_workers=num_workers)



    elif args.dataset=='celebA':
        size = 32
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CelebA(
            root = args.data_path,
            split = 'train',
            transform = transform_train,download=False)
        testset = CelebA(
            root = args.data_path,
            split = 'train',
            transform = transform_train,download=False)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=num_workers)

    else:
        raise NotImplementedError
    return trainloader,testloader, None


def get_UCI_data_loader(args, b_size,num_workers):
    p = load_data(args.dataset)

    train_data = p.data
    test_data = p.test_data
    valid_data = p.valid_data
    train_set = PrepareUCIData(train_data)
    test_set = PrepareUCIData(test_data)
    valid_set = PrepareUCIData(valid_data)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=b_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=b_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=b_size, shuffle=True, num_workers=num_workers)

    return trainloader,testloader,validloader

def get_toy_loader(args,b_size,device):
    import models.toy_models as tm
    N_samples = 5000
    dtype=  'float32'
    dataset = tm.BaseDataset(N_samples,dtype, device, b_size, args.data_path)

    params = {'batch_size': args.b_size,
          'shuffle': True,
          'num_workers': 0}
    return torch.utils.data.DataLoader(dataset, **params)

def get_data_loader(args, b_size,num_workers):
    if args.dataset_type=='images':
        trainloader,testloader, validloader = get_image_loader(args, b_size, num_workers)
        input_dims = None
    elif args.dataset_type=='UCI':
        trainloader,testloader,validloader = get_UCI_data_loader(args, b_size,num_workers)
        input_dims = np.array([trainloader.dataset.X.shape[1]])[0]
    elif args.dataset_type=='toy':
        trainloader = get_toy_loader(args, b_size,'cuda')
        input_dims = 3
        validloader = trainloader
        testloader = trainloader
    return trainloader,testloader,validloader, input_dims


# choose loss type
def get_loss(args):
    if args.criterion=='hinge':
        return cp.hinge
    elif args.criterion=='wasserstein':
        return cp.wasserstein
    elif args.criterion=='logistic':
        return cp.logistic
    elif args.criterion in ['kale', 'donsker']:
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


def get_normal(Z_dim, device):
    loc = torch.zeros(Z_dim).to(device)
    scale = torch.ones(Z_dim).to(device)
    normal = torch.distributions.normal.Normal(loc, scale)
    return torch.distributions.independent.Independent(normal,1)

class ConditionalNoiseGen(nn.Module):
    def __init__(self, truncation,device):
        super(ConditionalNoiseGen, self).__init__()
        self.truncation = truncation
        self.device = device
        self.num_classes = 1000
        labels = 1.*np.array(range(self.num_classes))/self.num_classes
        labels= torch.tensor(list(labels)).to(device)
        self.multinomial = torch.distributions.categorical.Categorical(labels)
    def log_prob(self,noise):
        Z,labels = noise
        prob = - 0.5 * torch.norm(Z, dim=1) ** 2
        return prob

    def sample(self,shape):
        Z = truncated_noise_sample(truncation=self.truncation, batch_size=shape[0])
        Z = torch.from_numpy(Z)
        Z = Z.to(self.device)
        labels = self.multinomial.sample(shape)
        return Z,labels

# return the distribution of the latent noise

def get_latent_sampler(args,potential,Z_dim,device):
    momentum = get_normal(Z_dim,device)
    if args.latent_sampler=='hmc':
        return samplers.HMCsampler(potential,momentum, T=args.num_sampler_steps, num_steps_min=10, num_steps_max=20,gamma=args.lmc_gamma,  kappa = args.lmc_kappa)
    elif args.latent_sampler=='lmc':
        return samplers.LMCsampler(potential,momentum, T=args.num_sampler_steps, num_steps_min=10, num_steps_max=20,gamma=args.lmc_gamma,  kappa = args.lmc_kappa)
    elif args.latent_sampler=='langevin':
        return samplers.LangevinSampler(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='zero_temperature_langevin':
        return samplers.ZeroTemperatureSampler(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='mala':
        return samplers.MALA(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='spherelangevin':
        return samplers.SphereLangevinSampler(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='dot':
        return samplers.DOT(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='trunclangevin':
        return samplers.TruncLangevinSampler(potential,momentum,trunc=args.trunc,  T=args.num_sampler_steps, num_steps_min=10, num_steps_max=20,gamma=args.lmc_gamma,  kappa = args.lmc_kappa)
    elif args.latent_sampler=='mh':
        return samplers.MetropolisHastings(potential, T=args.num_sampler_steps, gamma=args.lmc_gamma )
    elif args.latent_sampler=='imh':
        return samplers.IndependentMetropolisHastings(potential,  T=args.num_sampler_steps, gamma=args.lmc_gamma)


def get_latent_noise(args,dim,device):
    dim = int(dim)
    if args.latent_noise=='gaussian':
        loc = torch.zeros(dim).to(device)
        scale = torch.ones(dim).to(device)
        normal = torch.distributions.normal.Normal(loc, scale)
        return torch.distributions.independent.Independent(normal,1)
    elif args.latent_noise=='uniform':
        return torch.distributions.Uniform(torch.zeros(dim).to(device),torch.ones(dim).to(device))


# initialize neural net corresponding to type


# return a discriminator for the energy model
def get_energy(args,input_dims,device):
    if args.discriminator=='convolutional':
        return Discriminator(nn_type=args.d_model, bn=args.bn, no_trunc=args.no_trunc, skipinit=args.skipinit).to(device)
    elif args.discriminator=='nvp':
        return energy_model.NVP([input_dims], device,args.num_blocks,mode='discriminator',with_bn=args.dis_bn).to(device)
    elif args.discriminator=='made':
        return energy_model.MADEGenerator([input_dims],  mode='discriminator').to(device)
    elif args.discriminator=='maf':
        return energy_model.FlowGenerator([input_dims], device,args.num_blocks,'maf' , mode='discriminator',with_bn=args.dis_bn).to(device)
    elif args.discriminator=='mogmaf':
        return energy_model.FlowGenerator([input_dims], device,args.num_blocks,'mogmaf', mode='discriminator',with_bn=args.dis_bn).to(device)
    elif args.discriminator=='toy':
        return tm.Discriminator(3)
    elif args.discriminator=='are':
        return energy_model.Discriminator(input_dims, device).to(device)
    elif args.discriminator=='are4':
        return energy_model.Discriminator4(input_dims, device).to(device)

# return the base for the energy model
def get_base(args,input_dims,device):
    if args.generator == 'convolutional':
        net = Generator(nz=args.Z_dim, nn_type=args.g_model).to(device)
    elif args.generator =='gaussian':
        net = energy_model.GaussianGenerator([input_dims]).to(device)
    elif args.generator == 'made':
        net = energy_model.MADEGenerator([input_dims], mode='generator').to(device)
    elif args.generator == 'nvp':
        net = energy_model.NVP([input_dims], device,args.num_blocks,mode='generator', with_bn=args.gen_bn).to(device)
    elif args.generator == 'maf':
        net = energy_model.FlowGenerator([input_dims], device,args.num_blocks, 'maf' ,  mode='generator', with_bn=args.gen_bn).to(device)
    elif args.generator == 'mogmaf':
        net = energy_model.FlowGenerator([input_dims], device,args.num_blocks, 'mogmaf',  mode='generator', with_bn=args.gen_bn).to(device)
    elif args.generator == 'toy':
        net = tm.Generator(3)
    return net

def init_logs(args, run_id, log_dir):
    if args.save_nothing:
        return None, None, None
    os.makedirs(log_dir, exist_ok=True)

    samples_dir = os.path.join(log_dir,  f'samples_{run_id}_{args.slurm_id}')
    os.makedirs(samples_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(log_dir, f'checkpoints_{run_id}_{args.slurm_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)
                
    if args.log_to_file:
        log_file = open(os.path.join(log_dir, f'log_{run_id}_{args.slurm_id}.txt'), 'w', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file        
    
    # log the parameters used in this run
    with open(os.path.join(log_dir, f'params_{run_id}_{args.slurm_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    return log_dir,checkpoint_dir,samples_dir

def assign_device(device):
    if device >-1:
        device = 'cuda:'+str(device) if torch.cuda.is_available() and device>-1 else 'cpu'
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device

def load_dictionary(file_name):
    out_dict = {}
    with open(file_name) as f:
        for line in f:
            cur_dict = json.loads(line)
            keys = cur_dict.keys()
            for key in keys:
                if key in out_dict:
                    out_dict[key].append(cur_dict[key])
                else:
                    out_dict[key] = [cur_dict[key]]
    return out_dict
