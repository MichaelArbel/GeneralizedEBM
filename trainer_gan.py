import math

import torch
import torch.nn as nn

import numpy as np

# Plotting library.
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

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
        
        print(f"Process id: {str(os.getpid())} | hostname: {socket.gethostname()}")
        print(f"Run id: {self.run_id}")
        print(f"Time: {datetime.now()}")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(args))
        print('==> Building model..')

        self.with_fid = args.with_fid
        self.mode = args.mode
        self.build_model()
        
    # do the relevant stuff with the model
    def main(self):
        print(f'==> Mode: {self.mode}')
        if self.mode == 'train':
            self.train()
        elif self.mode == 'eval':
            self.evaluate_with_fid()
        elif self.mode == 'images':
            self.get_quick_images()
        elif self.mode == 'fids':
            self.get_fid_progress()
        

    # model building functions

    def build_model(self):
        self.train_loader, self.test_loader = hp.get_data_loader(self.args)
        self.generator = hp.get_net(self.args, 'generator', self.device)
        self.discriminator = hp.get_net(self.args, 'discriminator', self.device)

        # load models if path exists, define log partition if using kale and add to discriminator
        self.d_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        if self.args.g_path is not None:
            self.load_generator()
        if self.args.d_path is not None:
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





    #### FOR TRAINING THE NETWORK

    # just train the thing
    def train(self):
        # want epochs to start at 1
        for epoch in range(1, self.args.total_epochs + 1):
            print(f'Epoch: {epoch}')
            # only train specified network(s)
            if self.args.train_which == 'both':
                self.train_epoch()
            elif self.args.train_which == 'discriminator':
                self.train_discriminator_epoch()
            if np.mod(epoch, 10) == 0 or epoch == 1:
                self.evaluate_training()
            if np.mod(epoch, 5) == 0:
                self.save_checkpoint(epoch)

    # train for one epoch
    def train_epoch(self):
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
                print(f' gen loss (mean|std): {ag.mean()} | {ag.std()}, disc loss (mean|std): {ad.mean()} | {ad.std()}')
                accum_loss_g = []
                accum_loss_d = []

        if self.args.use_scheduler:
            self.scheduler_d.step()
            self.scheduler_g.step()

    # same as train_epoch, but shortcut to just train the discriminator
    def train_discriminator_epoch(self):
        accum_loss = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.d_loss = self.iteration(data, net_type='discriminator')
            accum_loss.append(self.d_loss.item())
            if batch_idx % 100 == 0 and batch_idx > 0:
                ad = np.asarray(accum_loss)
                print(f' critic loss (mean|std): {ad.mean()} | {ad.std()}')
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
        if self.args.criterion == 'kale':
            true_results = true_results + self.log_partition
            fake_results = fake_results + self.log_partition
        # calculate loss and propagate
        loss = self.loss(true_results, fake_results, net_type) 
        penalty = self.args.penalty_lambda * \
            cp.penalty_d(self.args, self.discriminator, data, fake_data, self.device)
        # print(penalty.item(), loss.item())
        total_loss = loss + penalty
        
        total_loss.backward()
        optimizer.step()
        return total_loss

    # save model parameters from a checkpoint, only used when training
    def save_checkpoint(self, epoch):
        if self.args.save_nothing:
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

    # just evaluate the performance (via KALE metric) during training
    def evaluate_training(self):
        self.discriminator.eval()
        self.generator.eval()
        kales, mean_fake, images, Zs = self.acc_stats(eval_posterior=False)
        self.sample_images()

        print(f'KALE: {kales["prior"]} (fake: {mean_fake["prior"]})')





    #### FOR CREATING IMAGES

    # just sample some images from the generator cus that's all we're here for
    def get_quick_images(self):
        print('==> Generating images from pre-trained model...')
        self.load_generator()
        self.load_discriminator()
        self.sample_images(epoch=self.args.seed)





    #### FOR EVALUATING PERFORMANCE WITH FID

    # evaluate a pretrained model thoroughly via FID
    def evaluate_with_fid(self, num_evaluations=4):
        print('==> Evaluating pre-trained model...')
        self.load_generator()
        self.load_discriminator()
        self.discriminator.eval()
        self.generator.eval()
        kales = {}
        fids = {}
        s_types = ['prior', 'posterior']
        for s in s_types:
            kales[s] = []
            fids[s] = []

        kf_path = os.path.join(self.log_dir, 'kales_and_fids.json')
        if os.path.isfile(kf_path):
            with open(kf_path, 'r') as f:
                kales, fids = json.load(f)
        else:
            for n in range(1, num_evaluations+1):
                print(f'\n\n====> EVALUATION #{n}')

                torch.manual_seed(n)
                np.random.seed(n)

                # do the normal evaluation, without fids
                i_kales, i_fakes, i_images, i_Zs = self.acc_stats(eval_posterior=True, eval_id=n)
                i_fids = {}
                for s in s_types:
                    # now calculate fids and print out the results
                    i_fid = cp.compute_fid(self.args, self.device, i_images[s], self.fid_model, self.train_loader, self.test_loader)
                    print(f'KALE-{s}: {i_kales[s]} (fake: {i_fakes[s]}), FID-{s}: {i_fid}')
                    kales[s].append(i_kales[s])
                    fids[s].append(i_fid)
                    
        # finally, calculate some summary statistics
        mean = np.array(kales['prior']).mean()
        std = np.array(kales['prior']).std()
        kales['kale_summary'] = {
            'mean': mean,
            'std': std
        }
        print(f'KALE-{s} | mean: {mean}, std: {std}')
        for s in s_types:
            these_fids = [i[1] for i in fids[s]]
            mean = np.array(these_fids).mean()
            std = np.array(these_fids).std()
            fids[f'{s}_summary'] = {
                'mean': mean,
                'std': std
            }
            print(f'FID-{s} | mean: {mean}, std: {std}')

        if not os.path.isfile(kf_path) and not self.args.save_nothing:
            with open(kf_path, 'w') as f:
                json.dump([kales, fids], f, indent=2)


    # calculate KALE and generated images
    def acc_stats(self, eval_posterior, eval_id=0):
        # change as necessary depending on the GPU
        bb_size = self.args.bb_size
        n_batches = int(self.args.fid_samples / bb_size) + 1
        # contribution of fake generated data
        mean_fake = {}
        images = {}
        fake_results = {}

        f_exists = False
        Zs = {}
        # only use posterior samples if we want them
        if eval_posterior:
            # check if we already have a Z we can just load so we don't have to compute everything
            fname = os.path.join(self.args.Z_folder, f'Z_eval_{eval_id}.pkl')
            if len(self.args.Z_folder) > 0 and os.path.isfile(fname):
                with open(fname, 'rb') as f:
                    Zs = pkl.load(f)
                    Z_keys = list(Zs)
                    f_exists = True
            s_types = ['prior', 'posterior']
        else:
            s_types = ['prior']
        
        # multiple batches, otherwise get memory issues
        print(f'==> Generating samples, with posterior sampling: {eval_posterior}')
        normal_gen = torch.distributions.Normal(torch.zeros((bb_size, self.args.Z_dim)).to(self.device),1)

        for s in s_types:
            print(f'Running type: {s}')
            m = 0
            avg_time = 0
            for b in range(n_batches):
                if b % 5 == 0:
                    print(f'  Starting batch {b+1}/{n_batches}, avg time {avg_time}s')
                if m < self.args.fid_samples:
                    # normally take batches of size bb_size, unless it's the last batch
                    bl = min(self.args.fid_samples - m, bb_size)
                    st = time.time()
                    
                    # if we want to generate posterior samples, here's where it happens
                    if not f_exists or s not in Z_keys:
                        prior_Z = normal_gen.sample()
                        if s == 'posterior':
                            posterior_Z = cp.sample_posterior(
                                prior_z=prior_Z,
                                g=self.generator,
                                h=self.discriminator,
                                device=self.device,
                                T=self.args.num_lmc_steps,
                                extract_every=0,
                                kappa=self.args.lmc_kappa,
                                gamma=self.args.lmc_gamma
                            )
                        else:
                            posterior_Z = prior_Z
                    else:
                        posterior_Z = Zs[s][m:m+bl].to(self.device)

                    # online calculating of avg time
                    et = time.time()
                    avg_time = avg_time*b/(b+1) + (et-st)/(b+1)

                    with torch.no_grad():
                        # could be small (because it's the last batch)
                        posterior_Z = posterior_Z[:bl]
                        fake_data = self.generator(posterior_Z)
                        results = self.discriminator(fake_data)
                        # save images to cpu because we have more memory there
                        if m == 0:
                            if not f_exists:
                                Zs[s] = torch.zeros([self.args.fid_samples]+list(posterior_Z.shape[1:]))
                            images[s] = torch.zeros([self.args.fid_samples]+list(fake_data.shape[1:]))
                            fake_results[s] = torch.zeros([self.args.fid_samples]+list(results.shape[1:]))

                        if not f_exists:
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

        # if we created posteriors, then save it because these are hard to generate
        if eval_posterior and not f_exists and not self.args.save_nothing:
            fname = os.path.join(self.samples_dir, f'Z_eval_{eval_id}.pkl')
            with open(fname, 'wb') as f:
                pkl.dump(Zs, f)

        return KALE, mean_fake, images, Zs




    #### FOR GETTING FID PROGRESS OVER LMC TIMESTEPS

    # for finding the FID scores for the same initial Z, over time
    def get_fid_progress(self):

        assert self.with_fid
        #fname = os.path.join(self.log_dir, f'posterior_data_{self.run_id}.pkl')

        total_extract_time = 300
        extract_every = 10

        avg_time = 0

        # if os.path.isfile(fname):
        #     with open(fname, 'rb') as f:
        #         images = pkl.load(f)
        #     print(f'Loaded existing images from {fname}')
        # else:
        bb_size = self.args.bb_size
        n_batches = int(self.args.fid_samples / bb_size) + 1

        m = 0
        normal_gen = torch.distributions.Normal(torch.zeros((bb_size, self.args.Z_dim)).to(self.device),1)
        images = []
        print('==> Evaluating successive FIDs')
        for b in range(n_batches):
            if b % 5 == 0:
                print(f'  Starting batch {b+1}/{n_batches}, avg time {avg_time}s')
            if m < self.args.fid_samples:
                bl = min(self.args.fid_samples - m, bb_size)
                st = time.time()
                prior_Z = normal_gen.sample()

                posterior_ts, posterior_Zs = cp.sample_posterior(
                    prior_z=prior_Z,
                    g=self.generator,
                    h=self.discriminator,
                    device=self.device,
                    T=total_extract_time,
                    extract_every=extract_every,
                    kappa=self.args.lmc_kappa,
                    gamma=self.args.lmc_gamma
                )
                et = time.time()
                avg_time = avg_time*b/(b+1) + (et-st)/(b+1)
                
                num_dps = len(posterior_Zs)
                with torch.no_grad():
                    for i in range(num_dps):
                        # could be small (because it's the last batch)
                        posterior_Z = posterior_Zs[i][:bl].to(self.device)
                        fake_data = self.generator(posterior_Z)
                        if b == 0:
                            images.append(torch.zeros([self.args.fid_samples]+list(fake_data.shape[1:])))
                        images[i][m:m+bl] = fake_data.cpu()
                    m += fake_data.size(0)

            # with open(fname, 'wb') as f:
            #     pkl.dump(images, f)
            #     print(f'Saved images into {fname}')

        fids = []
        for i, dp in enumerate(images):
            # save the posteriors
            fname = os.path.join(self.samples_dir, f'{self.run_id}_{str(i*extract_every).zfill(3)}_Z.pkl')
            with open(fname, 'wb') as f:
                pkl.dump(dp, f)
            # save the images themselves
            self.save_images(dp, name=f'{self.run_id}_{str(i*extract_every).zfill(3)}')
            fid = cp.compute_fid(self.args, self.device, dp, self.fid_model, self.train_loader, self.test_loader)
            fids.append(fid)
            print(F'FID at step {i*extract_every}: {fids[i]}')

        # save a json of the FIDs
        fids_train = [i[0] for i in fids]
        fids_test = [i[1] for i in fids]
        fids_formatted = [posterior_ts, fids_train, fids_test]
        with open(os.path.join(self.log_dir, f'posterior_fids_{self.run_id}.json'), 'w') as f:
            json.dump(fids_formatted, f, indent=2)
            print(f'Saved fids')




    #### HELPER FUNCTIONS USED SEVERAL TIMES, TO GENERATE IMAGES

    # samples 64 images, saves them
    def sample_images(self, epoch=0):
        if self.args.save_nothing:
            return
        self.discriminator.eval()
        self.generator.eval()
        normal_gen = hp.get_normal(self.args, self.device, 64)
        prior_z = normal_gen.sample()
        print(f'==> Producing 64 samples...')
        posterior_z = cp.sample_posterior(
            prior_z=prior_z,
            g=self.generator,
            h=self.discriminator,
            device=self.device,
            T=self.args.num_lmc_steps,
            extract_every=0,
            kappa=self.args.lmc_kappa,
            gamma=self.args.lmc_gamma
            )
        z_dic = {
            'prior': prior_z,
            'posterior': posterior_z
        }
        samples_dic = {}
        for key, z in z_dic.items():
            images = self.generator(z[:64])
            images_list = self.save_images(images, f'{str(epoch).zfill(3)}-{key}')
            samples_dic[key] = images_list

        # store sample image data so we can use it again later maybe
        if not self.args.save_nothing:
            with open(os.path.join(self.samples_dir, f'{str(epoch).zfill(3)}-data.pkl'), 'wb') as f:
                pkl.dump(samples_dic, f)

    # produces the actual np images and saves them
    def save_images(self, images, name):
        samples = images[:64].cpu().detach().numpy()
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)
        images_list = []
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            sample_t = sample.transpose((1,2,0)) * 0.5 + 0.5
            images_list.append(sample_t)
            plt.imshow(sample_t)

        plt.savefig(
            os.path.join(self.samples_dir, f'{name}.png'),
            bbox_inches='tight')
        plt.close(fig)
        return images_list





# helper functions

# make logging directories
def init_logs(args, run_id):
    if args.save_nothing:
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
    return log_dir,checkpoint_dir,samples_dir

def assign_device(device):
    if device >-1:
        device = 'cuda:'+str(device) if torch.cuda.is_available() and device>-1 else 'cpu'
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device
