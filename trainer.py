import math

import torch
import torch.nn as nn

import numpy as np

import csv
import sys
import os
import time
from datetime import datetime
import pprint
import socket
import json
import pickle as pkl
from torch.autograd import Variable
import pdb

import timeit

# Don't forget to select GPU runtime environment in Runtime -> Change runtime type

import helpers as hp
import compute as cp
import samplers
#from pytorch_pretrained_biggan import BigGAN
#from  models.generator import BigGANwrapper
from utils import timer

import models

from utils.fid_scheduler import FIDScheduler
from utils import vizualization as viz

class Trainer(object):
    def __init__(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.device = hp.assign_device(args.device)
        self.run_id = str(round(time.time() % 1e7))
        print(f"Run id: {self.run_id}")
        self.log_dir, self.checkpoint_dir, self.samples_dir = hp.init_logs(args, self.run_id, self.log_dir_formatter(args) )
        
        print(f"Process id: {str(os.getpid())} | hostname: {socket.gethostname()}")
        print(f"Run id: {self.run_id}")
        print(f"Time: {datetime.now()}")
        self.pp = pprint.PrettyPrinter(indent=4)
        self.pp.pprint(vars(args))
        print('==> Building model..')
        self.timer = timer.Timer()
        self.mode = args.mode
        self.build_model()

        

    # model building functions
    def log_dir_formatter(self,args):
            return os.path.join(args.log_dir, args.mode, args.dataset, 'temp_'+str(args.temperature))


    def build_model(self):
        self.train_loader, self.test_loader, self.valid_loader,self.input_dims = hp.get_data_loader(self.args)
        
        self.generator = hp.get_base(self.args, self.input_dims, self.device)
        self.discriminator = hp.get_energy(self.args,self.input_dims, self.device)
        self.noise_gen = hp.get_latent_noise(self.args,self.args.Z_dim, self.device)
        self.fixed_latents = self.noise_gen.sample([64])
        self.eval_latents =torch.cat([ self.noise_gen.sample([self.args.sample_b_size]).cpu() for b in range(int(self.args.fid_samples/self.args.sample_b_size)+1)], dim=0)
        self.eval_latents = self.eval_latents[:self.args.fid_samples]
        # load models if path exists, define log partition if using kale and add to discriminator
        self.d_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        if self.args.g_path is not None:
            self.load_generator()
            self.generator.eval()
        if self.args.d_path is not None:
            self.load_discriminator()
            self.discriminator.eval()

        else:
            if self.args.criterion == 'kale':
                self.log_partition = nn.Parameter(torch.zeros(1).to(self.device))
                self.d_params.append(self.log_partition)
            else:
                self.log_partition = Variable(torch.zeros(1, requires_grad=False)).to(self.device)

        if self.mode == 'train':
            # optimizers
            self.optim_d = hp.get_optimizer(self.args, 'discriminator', self.d_params)
            self.optim_g = hp.get_optimizer(self.args, 'generator', self.generator.parameters())
            self.optim_partition = hp.get_optimizer(self.args, 'discriminator', [self.log_partition])
            # schedulers
            self.scheduler_d = hp.get_scheduler(self.args, self.optim_d)
            self.scheduler_g = hp.get_scheduler(self.args, self.optim_g)
            self.scheduler_partition = hp.get_scheduler(self.args, self.optim_partition)
            self.loss = hp.get_loss(self.args)

            self.counter = 0
            self.g_counter = 0
            self.g_loss = torch.tensor(0.)
            self.d_loss = torch.tensor(0.)

        if self.args.latent_sampler in ['imh', 'dot','spherelangevin']:
            self.latent_potential = samplers.Independent_Latent_potential(self.generator,self.discriminator,self.noise_gen) 
        else:
            self.latent_potential = samplers.Latent_potential(self.generator,self.discriminator,self.noise_gen, self.args.temperature) 
        
        self.latent_sampler = hp.get_latent_sampler(self.args, self.latent_potential, self.device)
        if self.args.eval_fid:
            self.eval_fid = True
            print('==> Loading inception network...')
            block_idx = cp.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.fid_model = cp.InceptionV3([block_idx]).to(self.device)
            self.fid_scheduler = FIDScheduler(self.args)
            self.fid_scheduler.init_trainer(self)
            self.fid_train = -1.
        else:
            self.eval_fid = False

        dev_count = torch.cuda.device_count()    
        if self.args.dataparallel and dev_count>1 :
            self.generator = torch.nn.DataParallel(self.generator,device_ids=list(range(dev_count)))
            self.discriminator = torch.nn.DataParallel(self.discriminator,device_ids=list(range(dev_count)))

    def main(self):
        print(f'==> Mode: {self.mode}')
        if self.mode == 'train':
            self.train()
        elif self.mode == 'eval':
            self.eval()
        elif self.mode == 'sample':
            self.sample()

    def load_generator(self):
        g_model = torch.load(self.args.g_path, map_location=self.device)
        self.noise_gen = hp.get_normal(self.args, self.device)
        self.generator.load_state_dict(g_model)
        self.generator = self.generator.to(self.device)

    def load_discriminator(self):
        d_model = torch.load(self.args.d_path, map_location= self.device )
        self.discriminator.load_state_dict(d_model, strict=False)
        self.discriminator = self.discriminator.to(self.device)
        if self.args.criterion == 'kale':
            try:
                self.log_partition = d_model['log_partition'].to(self.device)
            except:
                self.log_partition = nn.Parameter(torch.zeros(1).to(self.device))


    #### FOR TRAINING THE NETWORK
    def train(self):
        done =False
        if self.args.initialize_log_partition:
            self.log_partition.data = self.init_log_partition()  
        while not done:
            self.train_epoch()
            if self.args.train_mode in ['both', 'base']:
                done =  self.g_counter >= self.args.total_gen_iter
            else:
                done =  self.counter >= self.args.total_gen_iter

    def train_epoch(self):

        accum_loss_g = []
        accum_loss_d = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach()
            self.counter += 1
            is_gstep, is_dstep = self.which_step()
            # discriminator takes n_iter_d steps of learning for each generator step
            if is_gstep:
                self.g_counter +=1
                self.g_loss = self.iteration(data, net_type='generator')
                accum_loss_g.append(self.g_loss.item())
            else:
                self.d_loss = self.iteration(data, net_type='discriminator')
                accum_loss_d.append(self.d_loss.item())
            
            if self.args.train_mode =='both':
                counter = self.g_counter
                is_valid_step = is_gstep
                if self.g_counter % self.args.disp_freq == 0 and is_gstep:
                    ag = np.asarray(accum_loss_g).mean()
                    ad = np.asarray(accum_loss_d).mean()
                    self.save_dictionary({'g_loss':ag, 'd_loss':ad, 'loss_iter': self.g_counter})
                    self.timer(self.g_counter, " base loss: %.8f, energy loss: %.8f" % ( ag, ad))
                    accum_loss_g = []
                    accum_loss_d = []

            elif self.args.train_mode =='base':
                counter = self.g_counter
                is_valid_step = is_gstep
                if self.g_counter % self.args.disp_freq == 0 and is_gstep:
                    ag = np.asarray(accum_loss_g).mean()
                    self.save_dictionary({'g_loss':ag, 'loss_iter': self.g_counter})
                    self.timer(self.g_counter, " base loss: %.8f" % ag)
                    accum_loss_g = []
            
            elif self.args.train_mode =='energy':
                counter = self.counter
                is_valid_step = is_dstep
                if self.counter % self.args.disp_freq == 0 and is_dstep:
                    ad = np.asarray(accum_loss_d).mean()
                    self.save_dictionary({'d_loss':ad, 'loss_iter': self.counter})
                    self.timer(self.counter, " energy loss: %.8f" % ad)
                    accum_loss_d = []

            if counter % self.args.checkpoint_freq == 0 and is_valid_step:
                if self.args.train_mode in ['both', 'base'] and self.args.dataset_type=='images':
                    images = self.sample_images(self.fixed_latents, self.args.sample_b_size)
                    viz.make_and_save_grid_images(images, f'Iter_{str(self.g_counter).zfill(3)}', self.samples_dir)
                self.save_checkpoint(self.g_counter)

            self.eval_fid = is_gstep and np.mod(self.g_counter, self.args.freq_fid)==0  and self.args.eval_fid
            self.eval_kale = is_valid_step and np.mod(counter,self.args.freq_kale)==0  and self.args.eval_kale
            self.eval()
            if self.args.use_scheduler:
                if self.eval_fid:
                    if eval_fid:
                        self.fid_scheduler.step(self.fid_train)
                else:
                    self.scheduler_d.step()
                    self.scheduler_g.step()
   
    def eval(self):
        if self.eval_fid or self.eval_kale:
            images = self.sample_images(self.eval_latents, self.args.sample_b_size, as_list=True)
            if self.eval_kale:
                KALE_train, _, base_mean, log_partition = self.compute_kale(self.train_loader, images)
                KALE_test, _, _ , _ = self.compute_kale(self.test_loader,  images, precomputed_stats = (base_mean,log_partition) )
                self.save_dictionary({'kale_train':KALE_train.item(), 'kale_test':KALE_test.item(), 'base_mean':base_mean.item(), 'log_partition':log_partition.item(), 'kale_iter':self.counter})
            if self.eval_fid:
            
                images = torch.split( torch.cat(images, dim=0), self.args.fid_b_size, dim=0)
                fid_train, fid_test = self.compute_fid( images, loader_types = ['train','valid'])
                self.fid_train = fid_train
                self.save_dictionary({'fid_train':fid_train, 'fid_test':fid_test, 'fid_iter':self.g_counter})


    def which_step(self):
        if self.args.train_mode =='both':
            if self.g_counter < 2 or  (self.g_counter%500==0) :
            #if self.counter == 0:
                n_iter_d = self.args.n_iter_d_init
            else:
                n_iter_d = self.args.n_iter_d
            is_gstep = (np.mod(self.counter, n_iter_d+1) == 0) and (self.counter > self.args.n_iter_d_init)
            return is_gstep, ~is_gstep
        elif self.args.train_mode =='base':
            return True, False
        elif self.args.train_mode =='energy':
            return False, True

    # take a step, and maybe train either the discriminator or generator. also used in eval
    def iteration(self, data, net_type, train_mode=True):
        optimizer = self.prepare_optimizer(net_type)
        # get data and run through discriminator
        Z = self.noise_gen.sample([self.args.noise_factor*data.shape[0]])
        with_gen_grad = train_mode and (net_type=='generator')
        with torch.set_grad_enabled(with_gen_grad):
            fake_data = self.generator(Z)

        with torch.set_grad_enabled(train_mode):
            true_results = self.discriminator(data)
            fake_results = self.discriminator(fake_data)
            log_partition = self.compute_log_partition(fake_results, net_type)

        if self.args.criterion in ['kale','donsker']:
            true_results = true_results + log_partition
            fake_results = fake_results + log_partition
        # calculate loss and propagate
        loss = self.loss(true_results, fake_results, net_type) 

        if train_mode:
            total_loss = self.add_penalty(loss, net_type, data, fake_data)
            total_loss.backward()
            self.grad_clip(optimizer, net_type=net_type)
            optimizer.step()

        return loss

    def prepare_optimizer(self,net_type):
        if net_type=='discriminator':           
            optimizer = self.optim_d
            self.discriminator.train()
            self.generator.eval()
        elif net_type=='generator':
            optimizer = self.optim_g
            self.generator.train()
            self.discriminator.eval()  
        optimizer.zero_grad()
        return optimizer

    def add_penalty(self,loss, net_type, data, fake_data):
        if net_type=='discriminator':
            penalty = self.args.penalty_lambda * cp.penalty_d(self.args, self.discriminator, data, fake_data, self.device)
            total_loss = loss + penalty
        else:
            total_loss = loss
        return total_loss

    def init_log_partition(self):
        log_partition = torch.tensor(0.).to(self.device)
        M = 0
        num_batches = 100
        self.generator.eval()
        self.discriminator.eval()
        for batch_idx in range(num_batches):
            with torch.no_grad():
                Z = self.noise_gen.sample([self.args.sample_b_size])
                fake_data = self.generator(Z)
                fake_data = -self.discriminator(fake_data)
                log_partition,M = cp.iterative_log_sum_exp(fake_data,log_partition,M)
        log_partition = log_partition - np.log(M)
        return torch.tensor(log_partition.item()).to(self.device)

    def init_log_partition(self):
        log_partition = torch.tensor(0.).to(self.device)
        M = 0
        num_batches = 100
        self.generator.eval()
        self.discriminator.eval()

        gen_loader = self.sample_images(self.eval_latents,self.args.noise_factor*self.args.b_size, as_list= True)
        for data in gen_loader:
            with torch.no_grad():
                fake_data = -self.discriminator(data.to(self.device))
                log_partition,M = cp.iterative_log_sum_exp(fake_data,log_partition,M)
        log_partition = log_partition - np.log(M)
        return torch.tensor(log_partition.item()).to(self.device)


    def compute_log_partition(self,fake_results, net_type):
        batch_log_partition = torch.logsumexp(-fake_results, dim=0)- np.log(fake_results.shape[0])
        batch_log_partition = batch_log_partition.squeeze()
        val_log_partition = self.log_partition

        if net_type=='discriminator':
            if self.args.criterion=='donsker':
                log_partition = batch_log_partition.detach()
            else:
                log_partition = val_log_partition
        else:
            log_partition = batch_log_partition

        return log_partition

    def grad_clip(self,optimizer, net_type='discriminator'):
        if net_type=='discriminator':
            params = self.d_params[:-1]
            for i, param in enumerate(params):
                new_grad = 2.*(param.grad.data)/(1+ (param.grad.data)**2)
                if math.isfinite(torch.norm(new_grad).item()):
                    param.grad.data = 1.*new_grad
                else:
                    print('nan grad')
                    param.grad.data = torch.zeros_like(new_grad)

            param = self.d_params[-1]
            new_grad = param.grad.data/(1-param.grad.data)
            if math.isfinite(torch.norm(new_grad).item()):
                param.grad.data = new_grad
            else:
                param.grad.data = torch.zeros_like(new_grad)

    #### FOR EVALUATING PERFORMANCEs

    # evaluate a pretrained model thoroughly via FID
    def compute_fid(self, gen_loader, loader_types = ['train','valid']):
        self.generator.to('cpu')
        self.discriminator.to('cpu')
        self.fid_model.to(self.device)

        fids = []
        pred_arr = cp.get_activations_from_loader(gen_loader, self.fid_model, self.device, self.args.fid_b_size )
        pred_arr = pred_arr.numpy()
        mu2 = np.mean(pred_arr, axis=0)
        sigma2 = np.cov(pred_arr, rowvar=False)
        for loader_type in loader_types:
            if loader_type == 'train':
                data_loader = self.train_loader
            elif loader_type == 'valid':
                data_loader = self.test_loader
            else:
                raise NotImplementedError

            mu1, sigma1 = cp.get_fid_stats(self.fid_model, data_loader, self.args.dataset, loader_type, self.device)            
            fid = cp.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            fids.append(fid)

        self.fid_model.to('cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        return fids

    def compute_kale(self,data_loader,base_loader, precomputed_stats=None):
        self.discriminator.eval()
        base_mean = torch.tensor(0.).to(self.device)
        data_mean = 0
        if precomputed_stats is None:
            M = 0
            with torch.no_grad():
                for img in base_loader:
                    energy  = -self.discriminator(img.to(self.device))
                    if self.args.criterion == 'donsker':
                        base_mean,M = cp.iterative_log_sum_exp(energy,base_mean,M)
                    else:
                        energy = -torch.exp(energy - self.log_partition ) 
                        base_mean, M = cp.iterative_mean(energy, base_mean,M)
            if self.args.criterion=='donsker':
                log_partition = 1.*base_mean -np.log(M)
                base_mean = torch.tensor(-1.).to(self.device)

            else:
                log_partition = self.log_partition
        else:
            base_mean, log_partition = precomputed_stats 

        M = 0
        for data, target in data_loader:
            with torch.no_grad():

                data_energy = -(self.discriminator(data.to(self.device)) + log_partition)
            data_mean, M = cp.iterative_mean(data_energy, data_mean,M)
        M=0
        if self.args.criterion=='donsker':
            data = base_loader[0]
            energy  = -self.discriminator(data.to(self.device))
            log_partition,M = cp.iterative_log_sum_exp(energy,torch.tensor(0.).to(self.device),M)
            log_partition -= np.log(M) 

        KALE = data_mean + base_mean + 1
        return KALE, data_mean, base_mean, log_partition

    #### FOR Sampling
    def init_latents(self):
        if self.args.latent_sampler == 'dot':
            priors = 1.*self.eval_latents.unsqueeze(-1).clone()
            out = torch.cat([self.eval_latents.unsqueeze(-1), priors ], dim=-1)
            return out
        else:
            return self.eval_latents

    def sample(self):
        T = 10
        max_saved = 50
        num_dps = int(self.args.num_sampler_steps/T)+1
        start = time.time()
        for i in range(num_dps):
            iter_num = i*T
            if i==0: 
                posteriors = self.init_latents()
            else:
                posteriors = self.sample_latents(posteriors, self.args.sample_b_size , T)
            images = self.sample_images(posteriors,self.args.fid_b_size, as_list=True)
            fid_train, fid_test = self.compute_fid( images, loader_types = ['train','valid'])
            images = torch.cat(images, dim=0)
            saved_images = images[:64]
            saved_posteriors = posteriors[:64]
            dic_arrays = {'images':saved_images.cpu().numpy(), 'latents':saved_posteriors.cpu().numpy()}
            self.save_dictionary({'fid_train':fid_train, 'fid_test':fid_test, 'temp':self.args.temperature}, dic_arrays=dic_arrays, index=iter_num)
            viz.make_and_save_grid_images(images, f'iter_{str(iter_num).zfill(3)}', self.samples_dir)
            end = time.time()
            print(F'FID at step {iter_num}: {fid_train},  avg time {end-start}')
            start = end
            if i%20==0 and i>0:
                self.latent_sampler.gamma *= 0.1
                print(f'decreasing lr for sampling: {self.latent_sampler.gamma}')

    def sample_latents(self,priors,b_size, T, with_acceptance = False):
        avg_time = 0
        posteriors = []
        avg_acceptences = []
        for b, prior in enumerate(priors.split(b_size, dim=0)):
            st = time.time()
            prior = prior.clone().to(self.device)
            posterior,avg_acceptence = self.latent_sampler.sample(prior,sample_chain=False,T=T)            
            posteriors.append(posterior)
            avg_acceptences.append(avg_acceptence)

        posteriors = torch.cat(posteriors, axis=0)
        avg_acceptences = np.mean(np.array(avg_acceptences), axis=0)

        if with_acceptance:
            return posteriors, avg_acceptences
        else:
            return posteriors

    def sample_images(self, latents, b_size =128, to_cpu = True, as_list=False):
        self.discriminator.eval()
        self.generator.eval()
        images = []
        for latent in latents.split(b_size, dim=0):
            with torch.no_grad():
                img = self.generator(latent.to(self.device))
            if to_cpu:
                img = img.cpu()
            images.append(img)
        if as_list:
            return images
        else:
            return torch.cat(images, dim=0)

### Savers

    def save_checkpoint(self, epoch,best=False):
        if self.args.save_nothing:
            return
        if self.args.train_mode in ['both', 'energy']:
            d_dict = self.discriminator.state_dict()
            if self.args.criterion == 'kale':
                d_dict['log_partition'] = self.log_partition
            if best:
                d_path = os.path.join(self.checkpoint_dir, f'd_best.pth')
            else:
                d_path = os.path.join(self.checkpoint_dir, f'd_{epoch}.pth')
            torch.save(d_dict, d_path)
            print(f'Saved {d_path}')
        if self.args.train_mode in ['both', 'base']:
            if best:
                g_path = os.path.join(self.checkpoint_dir, f'g_best.pth')
            else:    
                g_path = os.path.join(self.checkpoint_dir, f'g_{epoch}.pth')
            torch.save(self.generator.state_dict(), g_path)
            print(f'Saved {g_path}')

    # just evaluate the performance (via KALE metric) during training
    
    def save_dictionary(self,new_dict, dic_arrays = None ,index=0 ):
        if dic_arrays is not None:
            fname = os.path.join(self.samples_dir, f'MCMC_samples_{str(index).zfill(3)}.pkl')
            np.savez(fname, **dic_arrays)
            new_dict['index'] = index
            new_dict['path_arrays'] = fname
        file_name = os.path.join(self.samples_dir, f'stats_seed_{self.args.seed}')
        with open(file_name+'.json','a') as f:
            json.dump(new_dict,f)
            f.write(os.linesep)

class TrainerEBM(Trainer):
    def __init__(self, args):
        self.args = args
        self.train_loader, self.test_loader, self.valid_loader,self.input_dims = hp.get_data_loader(self.args)
        args.Z_dim = int(self.input_dims)
        self.dataset_size=  int(self.train_loader.dataset.X.shape[0])
        super(TrainerEBM, self).__init__(args)
        if self.args.combined_discriminator and self.args.criterion in ['kale','donsker']:
            self.discriminator = models.energy_model.CombinedDiscriminator(self.discriminator, self.generator)

        if self.args.criterion=='cd':
            sampler = hp.get_latent_sampler(self.args,self.discriminator,self.device)
            self.cd_sampler = samplers.ContrastiveDivergenceSampler(self.noise_gen, sampler, self.device)
    def log_dir_formatter(self,args):
            return os.path.join(args.log_dir, args.mode, args.dataset, args.discriminator, args.criterion)
    def select_statistics(self):
        statistics = ['nll_gen']
        has_log_density = getattr(self.discriminator, "log_density", None) is not None 
        has_log_partition =  getattr(self.discriminator, "log_partition", None) is not None 
        if has_log_density:
            statistics.append('nll_dis')
        if has_log_partition:
            statistics.append('gt_log_partition')
        if self.args.criterion =='cd':
            statistics.append('log_partition')
        elif self.args.criterion in ['kale', 'donsker']:
            statistics.append('kale')
        return statistics

    def eval(self):
        if np.mod(self.counter,10)==0:
            statistics = self.select_statistics()
            gen_loader = self.sample_images(self.eval_latents,self.args.noise_factor*self.args.b_size, as_list= True)

            train_dic,base_mean, log_partition = self.compute_stats_dic(self.train_loader, gen_loader, 'train' ,statistics)
            valid_dic,_,_ = self.compute_stats_dic(self.valid_loader,gen_loader,  'valid', statistics, precomputed_stats = (base_mean, log_partition))
            test_dic,_,_ = self.compute_stats_dic(self.test_loader,gen_loader,'test', statistics, precomputed_stats = (base_mean, log_partition))

            total_dic = {**train_dic,**valid_dic, **test_dic}
            out_dic = self.compute_final_stats(total_dic,statistics)
            self.save_dictionary(out_dic)
            print('Iteration:' +  str( int(self.counter)))
            self.pp.pprint(out_dic)
        #print(total_dic)
        # maybe keep track of best model on valid set
    def compute_final_stats(self,stat_dic, statistics):
        out = {}
        splits = ['train', 'valid', 'test']
        if self.args.combined_discriminator and self.args.criterion in  ['kale', 'donsker']:
            for split in splits:
                if 'nll_dis' in statistics and 'nll_gen' in statistics: 
                    out[split + '_nll']=  stat_dic[split +'_nll_gen'] + stat_dic[split +'_nll_dis']
                out[split+'_nkale'] = -stat_dic[split+'_kale'] + stat_dic[split +'_nll_gen']
                out[split+'_nkale_dist'] = -stat_dic[split+'_data_mean']  + stat_dic[split +'_nll_gen']
        elif  self.args.criterion in  ['cd', 'ml']:
            for split in splits:
                out[split+'_nll']= stat_dic[split+'_nll_dis']
        if 'gt_log_partition' in statistics:
            out['gt_log_partition'] = stat_dic['train_gt_log_partition']
        if self.args.criterion in  ['kale', 'donsker','cd']:
            out['log_partition'] = stat_dic['train_log_partition']
        return out

    def compute_stats_dic(self,data_loader, gen_loader, loader_type, statistics = ['nll_gen','nll_dis','kale'],precomputed_stats=None ):
        stats_dic = {}
        if 'nll_gen' in statistics:
            nll_gen = cp.compute_nll(data_loader,self.generator, self.device)
            stats_dic[loader_type+'_nll_gen'] = nll_gen.item()
        if 'nll_dis' in statistics:
            nll_gen = cp.compute_nll(data_loader,self.discriminator, self.device)
            stats_dic[loader_type+'_nll_dis'] = nll_gen.item()
        if 'kale' in statistics:

            KALE, data_mean, base_mean, log_partition = self.compute_kale(data_loader,gen_loader,precomputed_stats=precomputed_stats)
            stats_dic[loader_type+'_kale'] = KALE.item()
            stats_dic[loader_type+'_base_mean'] = base_mean.item()
            stats_dic[loader_type+'_data_mean'] = data_mean.item()
            stats_dic[loader_type+'_log_partition'] = log_partition.item()
            precomputed_stats = (base_mean, log_partition)
        elif 'log_partition' in statistics and self.args.criterion=='cd':
            stats_dic[loader_type+'_log_partition'] = self.cd_sampler.log_partition(self.args.noise_factor*self.args.b_size).item()


        if 'gt_log_partition' in statistics:
            stats_dic[loader_type+'_gt_log_partition'] = self.discriminator.log_partition().item()
        if precomputed_stats is None:


            return stats_dic, None, None
        else:
            base_mean, log_partition = precomputed_stats
            return stats_dic, base_mean, log_partition

    def iteration(self,data, net_type):
        if self.args.criterion in [ 'cd' ,'ml'] :
            return self.ebm_iteration(data, net_type)
        else:
            return super().iteration(data, net_type)

    def ebm_iteration(self,data,net_type='discriminator'):
        optimizer = self.prepare_optimizer(net_type)
        if self.args.criterion=='cd':
            gen_data = self.cd_sampler.sample(data.shape[0]*self.args.noise_factor)
            true_data = self.discriminator(data)
            fake_data = self.discriminator(gen_data)
            loss = true_data.mean() - fake_data.mean()
            total_loss = self.add_penalty(loss, net_type, data, gen_data)
        elif self.args.criterion=='ml':
            if net_type=='discriminator':
                model = self.discriminator
            elif net_type =='generator':
                model = self.generator
            loss = -model.log_density(data).mean()
            total_loss = self.add_penalty(loss, net_type, data, data)

        total_loss.backward()
        optimizer.step()
        return loss 

    def train_multiple_splits(self):
        for i, seed in enumerate(range(15)):
            print( ' Iteration : '+str(i) )
            self.args.seed = seed
            self.build_model()
            self.train()

class TrainerToy(Trainer):
    def __init__(self, args):
        self.train_loader, self.test_loader, self.valid_loader,self.input_dims = hp.get_data_loader(args)
        args.Z_dim = self.input_dims

        self.args = args
        super(TrainerToy, self).__init__(args)
    def log_dir_formatter(self,args):
            return os.path.join(args.log_dir, args.mode, args.dataset_type)  

    def eval(self):
        if np.mod(self.counter,self.train )==0: 
            out = self.compute_error_model()
            self.timer(self.counter, " energy error params: %.8f, base error params: %.8f" % (out['error_energy'],out['error_base'] ))

    def compute_error_model(self):
        params_energy = list(self.discriminator.parameters())
        true_params_energy = list(self.train_loader.dataset.energy.parameters())

        params_base = list(self.generator.parameters())
        true_params_base = list(self.train_loader.dataset.base.parameters())
        error_bases = np.array([ torch.norm(p-tp) for p,tp in zip(params_base, true_params_base)    ] ).sum()
        error_energy = np.array([ torch.norm(p-tp) for p,tp in zip(params_energy, true_params_energy)    ] ).sum()

        out ={'energy': params_energy,
               'base' : params_base,
               'true_energy': true_params_energy,
               'true_base': true_params_base,
               'error_energy': error_energy,
               'error_base':error_bases}
        #if not f_exists and not self.args.save_nothing:
        fname = os.path.join(self.samples_dir, f'saved_model_{self.counter}.pkl')
        with open(fname, 'wb') as f:
            pkl.dump(out, f)
        return out


