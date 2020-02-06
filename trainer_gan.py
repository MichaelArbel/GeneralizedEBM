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
import pickle as pkl

import pdb

import timeit

# Don't forget to select GPU runtime environment in Runtime -> Change runtime type

import helpers as hp
import compute as cp
    


class Trainer(object):
    def __init__(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.device = hp.assign_device(args.device)
        self.run_id = str(round(time.time() % 1e7))
        print(f"Run id: {self.run_id}")
        self.log_dir, self.checkpoint_dir, self.samples_dir = init_logs(args, self.run_id)
        
        print("Process id: " + str(os.getpid()) + " | hostname: " + socket.gethostname())
        print(f"Run id: {self.run_id}")
        print(f"Time: {datetime.now()}")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(args))
        print('==> Building model..')

        self.with_fid = args.with_fid
        self.mode = args.mode
        self.lmc_sample_size = args.lmc_sample_size
        self.sample_types = self.args.sample_types.split(',')
        self.build_model()
        
    # do the relevant stuff with the model
    def main(self):
        print(f'==> Mode: {self.mode}')
        if self.mode == 'train':
            self.train()
        elif self.mode == 'eval':
            self.eval_pre_trained()
        elif self.mode == 'images':
            self.get_quick_images()
        elif self.mode == 'fids':
            self.get_lmc_fids()
        

    # model building functions

    def build_model(self):
        self.train_loader, self.test_loader = hp.get_data_loader(self.args)
        self.generator = hp.get_net(self.args, 'generator', self.device)
        self.discriminator = hp.get_net(self.args, 'discriminator', self.device)

        # load models if path exists, define log partition and add to discriminator
        self.d_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        if len(self.args.g_path) > 0:
            self.load_generator()
        if len(self.args.d_path) > 0:
            self.load_discriminator()
            if self.args.criterion == 'kale':
                self.d_params.append(self.log_partition)
        else:
            if self.args.criterion == 'kale':
                self.log_partition = nn.Parameter(torch.zeros(1).to(self.device))
                self.d_params.append(self.log_partition)
            else:
                self.log_partition = torch.zeros(1, requires_grad=False).to(self.device)

        if self.mode == 'train':
            # optimizers
            self.optim_d = hp.get_optimizer(self.args, 'discriminator', self.d_params)
            self.optim_g = hp.get_optimizer(self.args, 'generator', self.generator.parameters())
            # schedulers
            self.scheduler_d = hp.get_scheduler(self.args, self.optim_d)
            self.scheduler_g = hp.get_scheduler(self.args, self.optim_g)

            self.loss = hp.get_loss(self.args)
            self.noise_gen = hp.get_normal(self.args, self.device, self.args.b_size)

            self.counter = 0
            self.g_loss = torch.tensor(0.)
            self.d_loss = torch.tensor(0.)

        if self.with_fid and self.mode != 'images':
            print('==> Loading inception network...')
            block_idx = cp.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.fid_model = cp.InceptionV3([block_idx]).to(self.device)

    def load_generator(self):
        g_model = torch.load(self.args.g_path)
        self.noise_gen = hp.get_normal(self.args, self.device, self.args.b_size)
        self.generator.load_state_dict(g_model)
        self.generator = self.generator.to(self.device)

    def load_discriminator(self):
        d_model = torch.load(self.args.d_path)
        self.discriminator.load_state_dict(d_model, strict=False)
        self.discriminator = self.discriminator.to(self.device)
        if self.args.criterion == 'kale':
            self.log_partition = d_model['log_partition']



    # model training functions

    # just train the thing
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

    # train for one epoch
    def train_epoch(self, epoch=0):
        if self.counter == 0:
            n_iter_d = self.args.n_iter_d_init
        else:
            n_iter_d = self.args.n_iter_d
        accum_loss_g = []
        accum_loss_d = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.counter += 1
            # discriminator takes n_iter_d steps of learning for each generator step
            if np.mod(self.counter, n_iter_d) == 0:
                self.g_loss = self.iteration(data, net_type='generator')
                accum_loss_g.append(self.g_loss.item())
            else:
                self.d_loss = self.iteration(data, net_type='discriminator')
                accum_loss_d.append(self.d_loss.item())
            if batch_idx % 100 == 0 and batch_idx > 0:
                ag = np.asarray(accum_loss_g)
                ad = np.asarray(accum_loss_d)
                print(f' gen loss: {ag.mean()} / {ag.std()}, disc loss: {ad.mean()} / {ad.std()}')
                accum_loss_g = []
                accum_loss_d = []

        if self.args.use_scheduler:
            self.scheduler_d.step()
            self.scheduler_g.step()

    # same as train_epoch, but shortcut to just train the discriminator
    def train_discriminator(self, epoch=0):
        accum_loss = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.d_loss = self.iteration(data, net_type='discriminator')
            accum_loss.append(self.d_loss.item())
            if batch_idx % 100 == 0 and batch_idx > 0:
                ad = np.asarray(accum_loss)
                print(f' critic loss: {ad.mean()} / {ad.std()}')
                accum_loss = []
        if self.args.use_scheduler:
            self.scheduler_d.step()

    # take a step, and maybe train either the discriminator or generator. also used in eval
    def iteration(self, data, net_type):
        if net_type=='discriminator':           
            optimizer = self.optim_d
            self.discriminator.train()
        elif net_type=='generator':
            optimizer = self.optim_g
            self.generator.train()
        optimizer.zero_grad()
        # get data and run through discriminator
        Z = self.noise_gen.sample()
        fake_data = self.generator(Z)
        true_results = self.discriminator(data)
        fake_results = self.discriminator(fake_data)
        if self.args.criterion=='kale':
            true_results = true_results + self.log_partition
            fake_results = fake_results + self.log_partition
        # calculate loss and propagate
        loss = self.loss(true_results, fake_results, net_type) 
        penalty = self.args.penalty_lambda * \
            cp.penalty_d(self.args, self.discriminator, data, fake_data, self.device)
        # print(penalty.item(), loss.item())
        total_loss = loss + penalty
        #print(penalty, loss)
        # if total_loss.detach().item() > 100:
        #     pdb.set_trace()
        
        total_loss.backward()
        if self.args.gradient_clip_norm != 0:
            if net_type == 'discriminator':
                nn.utils.clip_grad_norm_(self.d_params, self.args.gradient_clip_norm)
            elif net_type == 'generator':
                nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.gradient_clip_norm)
        optimizer.step()
        return total_loss


    # model evaluation functions

    # just sample some images from the generator cus that's all we're here for
    def get_quick_images(self):
        print('==> Generating images from pre-trained model...')
        self.load_generator()
        self.load_discriminator()
        self.sample_images(epoch=self.args.seed)

    # samples 64 images according to all types in the the sample_type argument, saves them
    def sample_images(self, epoch=0):
        if self.args.log_nothing:
            return
        self.discriminator.eval()
        self.generator.eval()
        normal_gen = hp.get_normal(self.args, self.device, 64)
        prior_z = normal_gen.sample()
        samples_dic = {}
        for s in self.sample_types:
            print(f'==> Producing 64 samples of type {s}...')
            posterior_z = cp.sample_posterior(
                prior_z=prior_z,
                s_type=s,
                g=self.generator,
                h=self.discriminator,
                device=self.device,
                n_samples=1,
                burn_in=80
                )
            samples = self.generator(posterior_z).cpu().detach().numpy()[:64]
            samples_dic[s] = []
            fig = plt.figure(figsize=(8, 8))
            gs = gridspec.GridSpec(8, 8)
            gs.update(wspace=0.05, hspace=0.05)
            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                sample_t = sample.transpose((1,2,0)) * 0.5 + 0.5
                samples_dic[s].append(sample_t)
                plt.imshow(sample_t)
            plt.savefig(
                os.path.join(self.samples_dir, f'{str(epoch).zfill(3)}-{s}.png'),
                bbox_inches='tight')
            plt.close(fig)
        # store sample image data so we can use it again later maybe
        with open(os.path.join(self.samples_dir, f'{str(epoch).zfill(3)}-data.pkl'), 'wb') as f:
            pkl.dump(samples_dic, f)

    # helper function meant for lmcs
    def save_images(self, images, epoch=0):
        samples = images[:64].cpu().detach().numpy()
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            sample_t = sample.transpose((1,2,0)) * 0.5 + 0.5
            plt.imshow(sample_t)
        plt.savefig(
            os.path.join(self.samples_dir, f'{str(epoch).zfill(3)}.png'),
            bbox_inches='tight')
        plt.close(fig)

    # evaluate a pretrained model thoroughly via FID
    def eval_pre_trained(self, num_evaluations=4):
        print('==> Evaluating pre-trained model...')
        self.load_generator()
        self.load_discriminator()
        self.discriminator.eval()
        self.generator.eval()
        kales = {}
        fids = {}
        for s in self.sample_types:
            kales[s] = []
            fids[s] = []

        kf_path = os.path.join(self.log_dir, 'kales_and_fids.json')
        if os.path.isfile(kf_path):
            with open(kf_path, 'r') as f:
                kales, fids = json.load(f)
        else:
            for n in range(num_evaluations):
                print(f'\n\n######### STARTING EVALUATION #{n}')
                kale, fid, _ = self.evaluate(eval_id=n)
                for s in self.sample_types:
                    kales[s].append(kale[s])
                    fids[s].append(fid[s])
            
        for s in self.sample_types:
            mean = np.array(kales[s]).mean()
            std = np.array(kales[s]).std()
            kales['SUMMARY'] = {
                'mean': mean,
                'std': std
            }
            print(f'{s}: KALE mean: {mean}, std: {std}')

            these_fids = [i[1] for i in fids[s]]
            mean = np.array(these_fids).mean()
            std = np.array(these_fids).std()
            fids['SUMMARY'] = {
                'mean': mean,
                'std': std
            }
            print(f'{s}: FID mean: {mean}, std: {std}')

        if not os.path.isfile(kf_path) and not self.args.log_nothing:
            with open(kf_path, 'w') as f:
                json.dump([kales, fids], f, indent=2)

    # see how well the network is doing and save the Zs because they are possibly expensive to compute
    def evaluate(self, eval_id=0):
        if self.mode == 'train':
            save_dir = self.checkpoint_dir
            Z_name = f'Z_{eval_id}.pkl'
            s_types = ['none']
            self.discriminator.eval()
            self.generator.eval()
        elif self.mode == 'eval':
            save_dir = self.samples_dir
            Z_name = f'Z_eval_{eval_id}.pkl'
            s_types = self.sample_types

        kales, mean_fake, images, Zs = self.acc_stats(s_types, eval_id)
        if not self.args.log_nothing:
            with open(os.path.join(save_dir, Z_name), 'wb') as f:
                pkl.dump(Zs, f)
        fids = {}
        for s in s_types:
            fids[s] = 'N/C'
            if self.with_fid:
                fids[s] = cp.compute_fid(self.args, self.device, images[s], self.fid_model, self.train_loader, self.test_loader)
            if s == 'none':
                print(f'KALE: {kales[s]} (fake: {mean_fake[s]}), FID: {fids[s]}')
            else:
                print(f'{s}-KALE: {kales[s]} (fake: {mean_fake[s]}), FID: {fids[s]}')
        return kales, fids, Zs


    def get_lmc_fids(self):

        #assert self.with_fid
        fname = os.path.join(self.log_dir, 'lmc_data.pkl')

        total_time = 100
        extract_every = 10

        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                images = pkl.load(f)
            print(f'Loaded existing images from {fname}')
        else:
            bb_size = self.args.bb_size
            lmc_sample_size = self.args.lmc_sample_size
            n_batches = int(self.args.fid_samples / bb_size / lmc_sample_size) + 1

            m = 0
            normal_gen = torch.distributions.Normal(torch.zeros((bb_size, self.args.Z_dim)).to(self.device),1)
            num_dps = int(total_time / extract_every)
            images = []
            for b in range(n_batches):
                if b % 5 == 0:
                    print(f'  Starting batch {b+1}/{n_batches}')
                if m < self.args.fid_samples:
                    bl = min(self.args.fid_samples - m, bb_size * lmc_sample_size)
                    prior_Z = normal_gen.sample()
                    posterior_Zs = cp.sample_posterior(
                        prior_z=prior_Z,
                        s_type='lmc', 
                        g=self.generator,
                        h=self.discriminator,
                        device=self.device,
                        n_samples=1,
                        burn_in=100,
                        extract_every=10
                    )
                    
                    with torch.no_grad():
                        for dp in range(num_dps):
                            # could be small (because it's the last batch)
                            posterior_Z = posterior_Zs[dp][:bl].to(self.device)
                            fake_data = self.generator(posterior_Z)
                            # save images to cpu because we have more memory there
                            if b == 0:
                                images.append(torch.zeros([self.args.fid_samples]+list(fake_data.shape[1:])))
                            images[dp][m:m+bl] = fake_data.cpu()
                        m += fake_data.size(0)

            with open(fname, 'wb') as f:
                pkl.dump(images, f)
                print(f'Saved images into {fname}')

        fids = []
        for i, dp in enumerate(images):
            self.save_images(dp, epoch=i*extract_every)
            fids.append(cp.compute_fid(self.args, self.device, dp, self.fid_model, self.train_loader, self.test_loader))
            print(F'FID at step {i*extract_every}: {fids[i]}')

        with open(os.path.join(self.log_dir, 'lmc_fids.pkl'), 'w') as f:
            json.dump(fids, f)
            print(f'Saved fids')




    # calculate KALE and generated images
    def acc_stats(self, s_types, eval_id=0):
        # change as necessary depending on the GPU
        bb_size = self.args.bb_size
        lmc_sample_size = self.args.lmc_sample_size
        n_batches = int(self.args.fid_samples / bb_size / lmc_sample_size) + 1

        # contribution of fake generated data
        mean_fake = {}
        images = {}
        # check if we already have a Z we can just load so we don't have to compute everything
        f_exists = False
        fname = os.path.join(self.args.Z_folder, f'Z_eval_{eval_id}.pkl')
        if len(self.args.Z_folder) > 0 and os.path.isfile(fname):
            with open(fname, 'rb') as f:
                Zs = pkl.load(f)
                Z_keys = list(Zs)
                f_exists = True
        else:
            Zs = {}
        fake_results = {}
        
        for s in s_types:
            # multiple batches, otherwise get memory issues
            print(f'==> Generating posterior samples, type: {s}')
            m = 0
            avg_time = 0
            normal_gen = torch.distributions.Normal(torch.zeros((bb_size, self.args.Z_dim)).to(self.device),1)
            for b in range(n_batches):
                if b % 5 == 0:
                    print(f'  Starting batch {b+1}/{n_batches}, avg time {avg_time}s')
                if m < self.args.fid_samples:
                    bl = min(self.args.fid_samples - m, bb_size * lmc_sample_size)
                    prior_Z = normal_gen.sample()
                    st = time.time()
                    if not f_exists or s not in Z_keys:
                        posterior_Z = cp.sample_posterior(
                            prior_z=prior_Z,
                            s_type=s, 
                            g=self.generator,
                            h=self.discriminator,
                            device=self.device,
                            n_samples=lmc_sample_size
                        )
                    else:
                        posterior_Z = Zs[s][m:m+bl].to(self.device)
                    et = time.time()
                    avg_time = avg_time*b/(b+1) + (et-st)/(b+1) # online calculating of avg time
                    with torch.no_grad():
                        # could be small (because it's the last batch)
                        posterior_Z = posterior_Z[:bl]
                        fake_data = self.generator(posterior_Z)
                        results = self.discriminator(fake_data)
                        # save images to cpu because we have more memory there
                        if m == 0:
                            images[s] = torch.zeros([self.args.fid_samples]+list(fake_data.shape[1:]))
                            Zs[s] = torch.zeros([self.args.fid_samples]+list(posterior_Z.shape[1:]))
                            fake_results[s] = torch.zeros([self.args.fid_samples]+list(results.shape[1:]))
                        Zs[s][m:m+bl] = posterior_Z.cpu()
                        images[s][m:m+bl] = fake_data.cpu()
                        fake_results[s][m:m+bl] = results.cpu()
                        m += fake_data.size(0)
            fake_results[s] = fake_results[s] + self.log_partition.cpu()
            mean_fake[s] = -torch.exp(-fake_results[s]).sum()
            mean_fake[s] /= m

        # contribution of real data
        with torch.no_grad():
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
        print(f'real: {mean_real} | lp: {self.log_partition.item()}')
        for s in s_types:
            KALE[s] = (mean_real + mean_fake[s] + 1).item()
            mean_fake[s] = mean_fake[s].item()

        return KALE, mean_fake, images, Zs



    # save model parameters from a checkpoint, only used when training
    def save_checkpoint(self, epoch):
        if self.args.log_nothing:
            return
        if self.args.train_which != 'generator':
            d_dict = self.discriminator.state_dict()
            if self.args.criterion == 'kale':
                d_dict['log_partition'] = self.log_partition
            d_path = os.path.join(self.checkpoint_dir, f'd_{epoch}.pth')
            torch.save(d_dict, d_path)
            print(f'Saved {d_path}')
        if self.args.train_which != 'discriminator':
            g_path = os.path.join(self.checkpoint_dir, f'g_{epoch}.pth')
            torch.save(self.generator.state_dict(), g_path)
            print(f'Saved {g_path}')
            



# helper functions

def init_logs(args, run_id):
    if args.log_nothing:
        return None, None, None
    log_name = args.log_name
    log_dir = os.path.join(args.log_dir, log_name, args.mode)
    os.makedirs(log_dir, exist_ok=True)

    samples_dir = os.path.join(log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    checkpoint_dir = None
    if args.mode == 'train':
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
                
    if args.log_to_file:
        log_file = open(os.path.join(log_dir, f'log_{run_id}.txt'), 'w', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file        
    
    # log the parameters used in this run
    with open(os.path.join(log_dir, f'params_{run_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)

    return log_dir, checkpoint_dir, samples_dir



