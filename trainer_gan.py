import math

#import tensorflow as tf
import torch
import torch.nn as nn

import numpy as np

# Plotting library.
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#import seaborn as sns

import csv
import sys
import os
import time
from datetime import datetime
import pprint
import socket
import json

import timeit

# Don't forget to select GPU runtime environment in Runtime -> Change runtime type

import helpers as hp
# fid_pytorch, inception
import metrics.fid_pytorch as fid_pytorch
from metrics.inception import InceptionV3


class Trainer(object):
    def __init__(self, args, load_inception=True):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.device = assign_device(args.device)
        now = datetime.now()
        self.trainer_id = now.strftime('%m-%d_%H-%M')
        self.log_dir,self.checkpoint_dir,self.samples_dir = make_log_dir(args, self.trainer_id)
        self.args.log_dir= self.log_dir
        if args.no_progress_bar:
            self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
            sys.stdout = self.log_file
            sys.stderr = self.log_file
        print("Process id: " + str(os.getpid()) + " | hostname: " + socket.gethostname())
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(args))
        print('==> Building model..')
        self.load_inception = load_inception
        self.build_model()

        

    # model building functions

    def build_model(self):      
        self.train_loader, self.test_loader = hp.get_data_loader(self.args)

        # discriminator
        self.discriminator = hp.get_net(self.args, 'discriminator', self.device)
        dis_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        if self.args.criterion=='kale':
            self.log_partition = nn.Parameter(torch.tensor(0.).to(self.device), requires_grad=True)
            dis_params.append(self.log_partition)
        else:
            self.log_partition = 0.
        # generator
        self.generator = hp.get_net(self.args, 'generator', self.device)
        # optimizers
        self.optim_d = hp.get_optimizer(self.args, dis_params)
        self.optim_g = hp.get_optimizer(self.args, self.generator.parameters())
        # schedulers
        self.scheduler_d = hp.get_scheduler(self.args, self.optim_d)
        self.scheduler_g = hp.get_scheduler(self.args, self.optim_g)

        self.loss = hp.get_loss(self.args)
        self.noise_gen = hp.get_latent(self.args,self.device)

        self.counter = 0
        self.g_loss = torch.tensor(0.)
        self.d_loss = torch.tensor(0.)

        if self.load_inception:
            print('==> Loading inception network...')
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.fid_model = InceptionV3([block_idx]).to(self.device)

    def load_generator(self):
        g_model = torch.load(self.args.g_path +'.pth')
        self.noise_gen = hp.get_latent(self.args,self.device)
        self.generator.load_state_dict(g_model)
        self.generator = self.generator.to(self.device)
    
    def load_discriminator(self):
        d_model = torch.load(self.args.d_path +'.pth')
        self.discriminator.load_state_dict(d_model)
        self.discriminator = self.discriminator.to(self.device)


    # model training functions

    # take a step, and maybe train either the discriminator or generator
    def iteration(self, data, net_type, train_mode=True):
        if train_mode:
            if net_type=='discriminator':
                #loss+= self.args.penalty_lambda*self.penalty_d(true_results,fake_results)            
                optimizer = self.optim_d
                self.discriminator.train()
            elif net_type=='generator':
                optimizer = self.optim_g
                self.generator.train()
            optimizer.zero_grad()
        else:
            self.generator.eval()
            self.discriminator.eval()
        # get data and run through discriminator
        Z = self.noise_gen.sample([self.args.b_size])
        gen_data = self.generator(Z)
        true_results = self.discriminator(data)
        fake_results = self.discriminator(gen_data)
        if self.args.criterion=='kale':
            true_results += self.log_partition
            fake_results += self.log_partition
        # calculate loss and propagate
        loss = self.loss(true_results,fake_results,net_type) 
        if train_mode:
            loss.backward()
            optimizer.step()
        return loss

    # train a particular epoch. epoch parameter not actually used...
    def train_epoch(self, epoch=0):
        if self.counter==0:
            n_iter_d = self.args.n_iter_d_init
        else:
            n_iter_d = self.args.n_iter_d
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.counter += 1
            # discriminator takes n_iter_d steps of learning for each generator step
            if np.mod(self.counter, n_iter_d) == 0:
                self.g_loss = self.iteration(data, net_type='generator', train_mode=True)
            else:
                self.d_loss = self.iteration(data, net_type='discriminator', train_mode=True)
            if batch_idx % 100 == 0:
                print(f'generator loss: {self.g_loss.item()}, critic loss: {self.d_loss.item()}')

    # same as train_epoch, but shortcut to just train the discriminator
    def train_discriminator(self, epoch=0):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.d_loss = self.iteration(data, net_type='discriminator', train_mode=True)
            if batch_idx % 100 == 0:
                print(f' critic loss: {self.d_loss.item()}')


    # just train the thing, both the generator and discriminator
    # which: which models to train?
    # pretrained: load pretrained models?
    def train(self, which='dg', pretrained=''):
        if 'd' in pretrained:
            self.load_discriminator()
        if 'g' in pretrained:
            self.load_generator()
        for epoch in range(self.args.total_epochs):
            print(f'Epoch: {epoch}')
            # only train specified network(s)
            if 'g' in which and 'd' in which:
                self.train_epoch(self, epoch)
            elif 'd' in which:
                self.train_discriminator(epoch)
            self.evaluate(epoch)
            self.sample_images(epoch)
            self.save_checkpoint(epoch)

    # model evaluation functions

    # fast, easy, model evaluation function to generate some images real fast, and maybe compute scores
    # evaluate a pretrained model. load generator and discriminator from a path
    def eval_pre_trained(self, evaluate=True):
        print('==> Evaluating pre-trained model...')
        self.load_generator()
        self.load_discriminator()
        self.sample_images(0)
        if self.eval_mode:
            self.evaluate(0)

    # samples 64 images according to all types in the the sample_type argument, saves them
    def sample_images(self, epoch=0):
        if np.mod(epoch, 10) == 0:
            sample_types = self.args.sample_type.split(',')
            for s in sample_types:
                sample_z = hp.get_latent_samples(self.args, self.device, s_type=s, g=self.generator, h=self.discriminator)
                samples = self.generator(sample_z).cpu().detach().numpy()[:64]

                fig = plt.figure(figsize=(8, 8))
                gs = gridspec.GridSpec(8, 8)
                gs.update(wspace=0.05, hspace=0.05)
                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)
                plt.savefig(self.samples_dir+'/{}-{}.png'.format(str(epoch).zfill(3),s), bbox_inches='tight')
                plt.close(fig)

    # evaluate a particular epoch, but only every 10
    def evaluate(self, epoch=0):
        if np.mod(epoch, 10) == 0:
            KALE, images = self.acc_stats()
            fid = 'N/A'
            if self.load_inception:
                fid = self.compute_fid(images)
            print(f'KALE: {KALE.item()}, FID: {fid}')

    # calculate KALE and generated images
    def acc_stats(self):
        n_batches = int(self.args.fid_samples/self.args.b_size)+1

        # mean losses for generated and real data
        mean_gen = 0.
        mean_data = 0.
        with torch.no_grad():
            m = 0
            for _ in range(n_batches):
                if m < self.args.fid_samples:
                    # create fake data and run through discriminator
                    Z = self.noise_gen.sample([self.args.b_size])
                    gen_data = self.generator(Z)
                    fake_results = self.discriminator(gen_data)
                    if self.args.criterion=='kale':
                        fake_results += self.log_partition
                    mean_gen += -torch.exp(-fake_results).sum()

                    lengthstuff= min(self.args.fid_samples-m,gen_data.shape[0])
                    if m == 0:
                        images = torch.zeros([self.args.fid_samples]+list(gen_data.shape[1:]))
                    images[m:m+lengthstuff,:]=gen_data[:lengthstuff,:].detach().cpu()
                    m += gen_data.size(0)
            mean_gen /= m
            m = 0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # get real data and run through discriminator
                data = data.to(self.device).clone().detach()
                true_data = self.discriminator(data)
                true_data += self.log_partition
                mean_data += -true_data.sum()
                m += true_data.size(0)
            mean_data /= m
            KALE = mean_data + mean_gen + 1

        return KALE, images

    # compute the FID score
    def compute_fid(self, images):
        print('==> Computing FID')

        mu2, sigma2 = fid_pytorch.compute_stats(images, self.fid_model,self.device,batch_size=128 )
        try:
            mu1_train, sigma1_train = hp.get_fid_stats(self.args.dataset+'_train')
        except:
            mu1_train, sigma1_train = self.compute_inception_stats('train',self.train_loader)
        fid_train = fid_pytorch.calculate_frechet_distance(mu1_train, sigma1_train, mu2, sigma2)

        if self.test_loader is not None:
            try:
                mu1_valid, sigma1_valid = hp.get_fid_stats(self.args.dataset+'_valid')
            except:
                mu1_valid, sigma1_valid = self.compute_inception_stats('valid',self.test_loader)
            fid_valid = fid_pytorch.calculate_frechet_distance(mu1_valid, sigma1_valid, mu2, sigma2)

            return fid_train,fid_valid
            
        # um do something about this
        return fid_score

    def compute_inception_stats(self,type_dataset,data_loader):
        mu_train,sigma_train= fid_pytorch.compute_stats_from_loader(self.fid_model,data_loader,self.device)
        #mu_1,sigma_1 = get_fid_stats(self.args.dataset+'_'+type_dataset)
        path = 'metrics/res/stats_pytorch/fid_stats_'+self.args.dataset+'_'+type_dataset+'.npz'
        np.savez(path, mu=mu_train, sigma=sigma_train)
        #if self.test_loader is not None:
        #   mu_test,sigma_test= fid_pytorch.compute_stats_from_loader(self.fid_model,self.test_loader,self.device)
        return mu_train,sigma_train

    def save_checkpoint(self,epoch):
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, f'disc_{epoch}'))
        torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, f'gen_{epoch}'))
        if self.args.criterion=='kale':
            torch.save({'log_partition':self.log_partition}, os.path.join(self.checkpoint_dir, f'log_partition_{epoch}' ))

def make_log_dir(args, trainer_id):
    if args.with_sacred:
        log_dir = args.log_dir + '_' + args.log_name
    else:
        log_dir = os.path.join(args.log_dir,args.log_name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    checkpoint_dir = os.path.join(log_dir,'checkpoints')
    samples_dir = os.path.join(log_dir,'samples')
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(samples_dir):
        os.mkdir(samples_dir)
    with open(os.path.join(log_dir, 'params.json'), 'w', encoding='utf-8') as f:
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

