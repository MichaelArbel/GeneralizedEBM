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
# from datetime import datetime
import pprint
import socket
import json
import pickle as pkl

import pdb

import timeit

# Don't forget to select GPU runtime environment in Runtime -> Change runtime type

import helpers as hp
# fid_pytorch, inception
import metrics.fid_pytorch as fid_pytorch
from metrics.inception import InceptionV3

from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer(object):
    def __init__(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.device = assign_device(args.device)
        # now = datetime.now()
        # self.trainer_id = now.strftime('%m-%d_%H-%M')
        self.log_dir, self.checkpoint_dir, self.samples_dir = make_log_dir(args)
        if args.log_to_file:
            self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
            sys.stdout = self.log_file
            sys.stderr = self.log_file
        print("Process id: " + str(os.getpid()) + " | hostname: " + socket.gethostname())
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(args))
        print('==> Building model..')
        self.with_fid = args.with_fid
        self.sample_types = self.args.sample_type.split(',')
        self.build_model()

        self.mode = args.mode

    def main(self):
        print(f'==> Mode: {self.args.mode}')
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'eval':
            self.eval_pre_trained()
        elif self.args.mode == 'images':
            self.get_quick_images()
        

    # model building functions

    def build_model(self):      
        self.train_loader, self.test_loader = hp.get_data_loader(self.args)
        # generator
        self.generator = hp.get_net(self.args, 'generator', self.device)
        # discriminator
        self.discriminator = hp.get_net(self.args, 'discriminator', self.device)

        # load models if path exists, define log partition
        dis_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        if len(self.args.g_path) > 0:
            self.load_generator()
        if len(self.args.d_path) > 0:
            self.load_discriminator()
            if self.args.criterion == 'kale':
                dis_params.append(self.log_partition)
        else:
            if self.args.criterion == 'kale':
                self.log_partition = nn.Parameter(torch.zeros(1).to(self.device), requires_grad=True)
                dis_params.append(self.log_partition)
            else:
                self.log_partition = 0.

        if self.mode == 'train':
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

        if self.with_fid:
            print('==> Loading inception network...')
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.fid_model = InceptionV3([block_idx]).to(self.device)

    def load_generator(self):
        g_model = torch.load(self.args.g_path)
        self.noise_gen = hp.get_latent(self.args,self.device)
        self.generator.load_state_dict(g_model)
        self.generator = self.generator.to(self.device)
    
    def load_discriminator(self):
        d_model = torch.load(self.args.d_path)
        self.discriminator.load_state_dict(d_model, strict=False)
        self.discriminator = self.discriminator.to(self.device)
        if self.args.criterion == 'kale':
            self.log_partition = d_model['log_partition']



    # model training functions

    # take a step, and maybe train either the discriminator or generator
    def iteration(self, data, net_type):
        if self.mode == 'train':
            if net_type=='discriminator':           
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
        fake_data = self.generator(Z)
        true_results = self.discriminator(data)
        fake_results = self.discriminator(fake_data)
        if self.args.criterion=='kale':
            true_results = true_results + self.log_partition
            fake_results = fake_results + self.log_partition
        # calculate loss and propagate
        loss = self.loss(true_results, fake_results, net_type) 
        penalty = self.args.penalty_lambda * self.penalty_d(data, fake_data)
        total_loss = loss + penalty
        if self.mode == 'train':
            total_loss.backward()
            optimizer.step()
        return total_loss

    def _gradient_penalty(self, true_data, fake_data):
        batch_size = true_data.size()[0]
        size_inter = min(batch_size,fake_data.size()[0])
        # Calculate interpolation
        alpha = torch.rand(size_inter,1,1,1)
        alpha = alpha.expand_as(true_data)
        alpha = alpha.to(self.device)
        
        interpolated = alpha*true_data.data[:size_inter] + (1-alpha)*fake_data.data[:size_inter]
        #interpolated = torch.cat([true_data.data,fake_data.data],dim=0)
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sum(gradients ** 2, dim=1)

        # Return gradient penalty
        return gradients_norm.mean()


    def penalty_d(self, true_data, fake_data):
        penalty = 0.
        len_params = 0.
        if self.args.penalty_type=='l2':
            for params in self.discriminator.parameters():
                penalty += torch.sum(params**2)
        elif self.args.penalty_type=='gradient':
            penalty = self._gradient_penalty(true_data, fake_data)
        elif self.args.penalty_type=='gradient_l2':
            for params in self.discriminator.parameters():
                penalty += torch.sum(params**2)
                len_params += np.sum(np.array(list(params.shape)))
            penalty = penalty/len_params
            g_penalty = self._gradient_penalty(true_data, fake_data)
            #ratio = penalty/(g_penalty+1e-6)
            #ratio = ratio.detach()
            p = penalty.detach().item()
            gp = g_penalty.detach().item()
            penalty += g_penalty
        return penalty

    # train a particular epoch. epoch parameter not actually used...
    def train_epoch(self, epoch=0):
        if self.counter==0:
            n_iter_d = self.args.n_iter_d_init
        else:
            n_iter_d = self.args.n_iter_d
        accum_loss_g = 0
        accum_loss_d = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.counter += 1
            # discriminator takes n_iter_d steps of learning for each generator step
            if np.mod(self.counter, n_iter_d) == 0:
                self.g_loss = self.iteration(data, net_type='generator')
                accum_loss_g += self.g_loss.item()
            else:
                self.d_loss = self.iteration(data, net_type='discriminator')
                accum_loss_d += self.d_loss.item()
            if batch_idx % 100 == 99:
                print(f'generator loss: {accum_loss_g/100}, critic loss: {accum_loss_d/100}')
                accum_loss_g = 0
                accum_loss_d = 0

    # same as train_epoch, but shortcut to just train the discriminator
    def train_discriminator(self, epoch=0):
        accum_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.d_loss = self.iteration(data, net_type='discriminator')
            accum_loss += self.d_loss.item()
            if batch_idx % 100 == 99:
                print(f' critic loss: {accum_loss/100}')
                accum_loss = 0


    # just train the thing, both the generator and discriminator
    # which: which models to train?
    # pt: load pretrained models?
    def train(self):
        if len(self.args.d_path) > 0:
            self.load_discriminator()
        if len(self.args.g_path) > 0:
            self.load_generator()
        for epoch in range(self.args.total_epochs):
            print(f'Epoch: {epoch}')
            # only train specified network(s)
            if self.args.train_which == 'both':
                self.train_epoch(epoch)
            elif self.args.train_which == 'discriminator':
                self.train_discriminator(epoch)
            if np.mod(epoch, 10) == 0:
                self.evaluate(epoch)
                self.sample_images(epoch)
            if np.mod(epoch, 2) == 0:
                self.save_checkpoint(epoch)

    # model evaluation functions

    # fast, easy, model evaluation functions to generate some images real fast, and compute scores
    # evaluate a pretrained model. load generator and discriminator from a path
    def eval_pre_trained(self, num_evaluations=1, save_Z=True):
        print('==> Evaluating pre-trained model...')
        self.load_generator()
        self.load_discriminator()
        kales = {}
        fids = {}
        for s in self.sample_types:
            kales[s] = []
            fids[s] = []
        for n in num_evaluations:
            kale, fid, _ = self.evaluate(eval_id=f'eval_{n}')
            for s in self.sample_types:
                kales[s].append(kale[s])
                fids[s].append(fid[s])
        
        with open(os.path.join(self.log_dir, 'kales_and_fids.json'), 'w') as f:
            json.dump([kales, fids], f, indent=4)

        for s in self.sample_types:
            mean = np.array(kales[s]).mean()
            std = np.array(kales[s]).std()
            print(f'KALE mean: {mean}, std: {std}')

            mean = np.array(fids[s]).mean()
            std = np.array(fids[s]).std()
            print(f'FID mean: {mean}, std: {std}')


    def get_quick_images(self):
        print('==> Generating images from pre-trained model...')
        self.load_generator()
        self.load_discriminator()
        self.sample_images(epoch=0)

    # samples 64 images according to all types in the the sample_type argument, saves them
    def sample_images(self, epoch=0):
        normal_gen = torch.distributions.Normal(torch.zeros((self.args.b_size, self.args.Z_dim)).to(self.device),1)
        prior_z = normal_gen.sample()
        for s in self.sample_types:
            print(f'==> Producing samples of type {s}...')
            sample_z = hp.get_latent_samples(prior_z=prior_z, s_type=s, g=self.generator, h=self.discriminator)
            samples = self.generator(sample_z).cpu().numpy()[:64]

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
    def evaluate(self, eval_id=0):
        KALE, mean_fake, images, Zs = self.acc_stats(save_Z_id=eval_id)
        for s in self.sample_types:
            fid = 'N/C'
            if self.with_fid:
                fid = self.compute_fid(images[s])
            if s == 'none':
                print(f'KALE: {KALE[s].item()} (fake: {mean_fake[s].item()}, FID: {fid}')
            else:
                print(f'{s}-KALE: {KALE[s].item()} (fake: {mean_fake[s].item()}, FID: {fid}')
        return KALE, fid, Zs


    # calculate KALE and generated images
    def acc_stats(self):
        bb_size = 1000
        n_batches = int(self.args.fid_samples / bb_size) + 1

        # contribution of fake generated data
        mean_fake = {}
        images = {}
        Zs = {}

        if self.mode == 'train':
            s_types = ['none']
        else:
            s_types = self.sample_types
        
        for s in s_types:
            # multiple batches, otherwise get memory issues
            print(f'==> Generating posterior samples, type: {s}')
            m = 0
            normal_gen = torch.distributions.Normal(torch.zeros((bb_size, self.args.Z_dim)).to(self.device),1)
            for b in range(n_batches):
                print('\rStarting batch {b}/{n_batches}')
                if m < self.args.fid_samples:
                    prior_z = normal_gen.sample()
                    fake_data_mle_Z = hp.get_latent_samples(
                        prior_z=prior_z, s_type=s, 
                        g=self.generator, h=self.discriminator, sampler=normal_gen
                    )
                    with torch.no_grad():
                        # could be small (because it's the last batch)
                        bl = min(self.args.fid_samples - m, bb_size)

                        fake_data = self.generator(fake_data_mle_Z)
                        # save images to cpu
                        if m == 0:
                            images[s] = torch.zeros([self.args.fid_samples]+list(fake_data.shape[1:]))
                            Zs[s] = torch.zeros([self.args.fid_samples]+list(fake_data_mle_Z.shape[1:]))
                        Zs[s][m:m+bl] = fake_data_mle_Z.cpu()
                        images[s][m:m+bl] = fake_data.detach().cpu()
                        m += fake_data.size(0)

            fake_results = self.discriminator(fake_data) + self.log_partition
            mean_fake[s] = -torch.exp(-fake_results)
            mean_fake[s] /= m
            
        # contribution of real data
        with torch.no_grad:
            m = 0
            mean_real = 0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # get real data and run through discriminator
                data = data.to(self.device).detach()
                real_results = self.discriminator(data) + self.log_partition
                mean_real += -real_results.sum()
                m += real_results.size(0)
            mean_real /= m

        KALE = {}
        print(f'real: {mean_real} | lp: {self.log_partition}')
        for s in self.sample_types:
            KALE[s] = mean_real + mean_fake[s] + 1

        return KALE, mean_fake, images, Zs


        # n_batches = int(self.args.fid_samples/self.args.b_size)+1
        # with torch.no_grad():
        #     m = 0
        #     # for _ in range(n_batches):
        #     #     if m < self.args.fid_samples:
        #     #         # create fake data and run through discriminator
        #     #         Z = self.noise_gen.sample([self.args.b_size])
        #     #         fake_data = self.generator(Z)
        #     #         # fake_data_mle_Z = hp.get_latent_samples(self.args, self.device, s_type='lmc', g=self.generator, h=self.discriminator)
        #     #         # fake_data = self.generator(fake_data_mle_Z)
        #     #         # fake_results = self.discriminator(fake_data)
        #     #         if self.args.criterion == 'kale':
        #     #             fake_results = fake_results + self.log_partition
        #     #         mean_gen += -torch.exp(-fake_results).sum()

        #     #         lengthstuff= min(self.args.fid_samples-m,fake_data.shape[0])
        #     #         if m == 0:
        #     #             images = torch.zeros([self.args.fid_samples]+list(fake_data.shape[1:]))
        #     #         images[m:m+lengthstuff,:]=fake_data[:lengthstuff,:].detach().cpu()
        #     #         m += fake_data.size(0)
        #     # mean_gen /= m
        #     m = 0
        #     for batch_idx, (data, target) in enumerate(self.test_loader):
        #         # get real data and run through discriminator
        #         data = data.to(self.device).detach()
        #         real_results = self.discriminator(data) + self.log_partition
        #         mean_data += -real_results.sum()
        #         m += real_results.size(0)
        #     mean_data /= m
        #     KALE = mean_data + mean_gen + 1
            
        #     real_data_mean = (real_results - self.log_partition).mean()
        #     print(f'real mean: {real_data_mean}')
        #     for s in sample_types:
        #         fake_data_mean = (fake_results - self.log_partition).mean()
        #         print(f'fake-{s} mean: {fake_data_mean}')
                
            # pdb.set_trace()
            # ...

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

    # save model parameters from a checkpoint
    def save_checkpoint(self, epoch):
        if self.args.train_which != 'generator':
            d_dict = self.discriminator.state_dict()
            if self.args.criterion == 'kale':
                d_dict['log_partition'] = self.log_partition
            torch.save(d_dict, os.path.join(self.checkpoint_dir, f'd_{epoch}.pth'))
        if self.args.train_which != 'discriminator':
            torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, f'g_{epoch}.pth'))
        



# helper functions

def make_log_dir(args, trainer_id=0):
    if args.with_sacred:
        log_dir = args.log_dir + '_' + args.log_name
    else:
        log_dir = os.path.join(args.log_dir,args.log_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir,'checkpoints')
    samples_dir = os.path.join(log_dir,'samples')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    # log the parameters used in this run
    with open(os.path.join(log_dir, 'run_params.json'), 'w', encoding='utf-8') as f:
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

