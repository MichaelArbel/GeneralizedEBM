from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import sonnet as snt

import numpy as np

# Plotting library.
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow_probability as tfp
import bisect

tfd = tfp.distributions
tfb = tfp.bijectors
from enum import Enum

from collections.abc import Iterable
import numpy as np
import utils

import sys
assert sys.version_info >= (3, 6), "Sonnet 2 requires Python >= 3.6"
import csv
import os
import time
import numpy as np
import pprint
import socket
import pickle
# Don't forget to select GPU runtime environment in Runtime -> Change runtime type

from copy import deepcopy
import metrics.fid_official_tf as fid
import tensorflow as tf

class Trainer(object):
	def __init__(self,args):
		torch.manual_seed(args.torch_seed)
		np.random.seed(args.numpy_seed)
		self.args = args
		self.device = assign_device(args.device)    
		self.log_dir = make_log_dir(args)
		self.args.log_dir= self.log_dir
		self.make_checkpoint()
		
		if args.no_progress_bar:
			self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
			sys.stdout = self.log_file
			sys.stderr = self.log_file
		print("Process id: " + str(os.getpid()) + " | hostname: " + socket.gethostname())
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(args))
		print('==> Building model..')
		self.build_model()


	def build_model(self):		
		self.train_loader, self.test_loader = get_data_loader(self.args)
		self.discriminator = get_net(self.args, 'discriminator', self.device)
		self.generator = get_net(self.args, 'generator', self.device)
		trainable_params = list(filter(lambda p: p.requires_grad, discriminator.parameters()))
		self.optim_d = get_optimizer(self.args,trainable_params)
		self.optim_g = get_optimizer(self.args,self.generator.parameters())
		self.scheduler_d = get_scheduler(self.args, self.optim_d)
		self.scheduler_g = get_scheduler(self.args, self.optim_g)
		self.loss = get_loss(args)
		self.noise_gen = get_latent(self.args,self.device)
		self.fixed_z = Variable(self.noise_gen([self.args.b_size]))
		self.penalty_d = get_penatly(self.args)
		self.counter =0

	def iteration(self,data,loss_type,train_mode='train'):
		if train_mode=='train':
			self.optim_d.zero_grad()
			self.optim_g.zero_grad()
		Z = self.noise_gen.sample([self.args.b_size])
		gen_data = self.generator(Z)
		true_data = self.discriminator(data)
		fake_data = self.discriminator(gen_data)
		loss = self.loss(true_data,fake_data,loss_type) 
		if loss_type=='discriminator':
			loss+= self.args.penalty_lambda*self.penalty_d(true_data,fake_data)			
			optimizer = self.optim_d
		elif loss_type=='generator':
			optimizer = self.optim_g
		if train_mode=='train':
			loss.backward()
			optimizer.step()
		return loss

	def train(self,epoch):
		if self.counter==0:
			n_iter_d = self.args.n_iter_d_init
		else:
			n_iter_d = self.args.n_iter_d
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data = Variable(data.to(self.device))
			self.counter += 1
			if np.mod(self.counter, n_iter_d):
				g_loss = self.iteration(data,'generator')
				if batch_idx % 100 == 0:
            		print('gen loss', g_loss[0])
			else:
				d_loss = self.iteration(data,'discriminator')
				if batch_idx % 100 == 0:
            		print('critic loss', d_loss[0])


	def sample_images(self,epoch):

	    samples = self.generator(self.fixed_z).cpu().detach().numpy()[:64]
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
	    if not os.path.exists('out/'):
	        os.makedirs('out/')
	    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
	    plt.close(fig)

	def evaluate(self,epoch):
		Kale,images = self.acc_stats()
		fid = self.compute_fid(images)
		print('Kale ' +  str(Kale) + ', FID: '+ str(fid))

	def save(self, epoch):

		


	def acc_stats(self):
	    n_batches = int(self.args.fid_samples/self.args.b_size)+1

	    mean_gen = 0.
	    mean_data = 0.
	    with torch.no_grad():
	   		m = 0
		    for _ in range(n_batches):
		    	if m<self.args.fid_samples:
		    		Z = self.noise_gen.sample([self.args.b_size])
					gen_data = self.generator(Z)
					fake_data = self.discriminator(gen_data)
					mean_gen += -torch.exp(-fake_data).sum()
					# rescale images to [0,255]  assuming images range btween -1 and 1 
					gen_data = gen_data*127.5+127.5
					lengthstuff= min(sample_size-m,gen_data.shape[0])
					if m==0:
						images = np.zeros([self.args.fid_samples]+[gen_data.shape[1:]])
					images[m:m+lengthstuff,:]=gen_data[:lengthstuff,:].detach().cpu().numpy()
		            m = m + outputs.size(0)
		    mean_gen /=  m
		    m = 0
			for batch_idx, (data, target) in enumerate(self.test_loader):
				data = Variable(data.to(self.device))
				true_data = self.discriminator(data)
				mean_data += -true_data.sum()
				m += true_data.size(0)
			mean_data /= m
			Kale = mean_data + mean_gen + 1

		return Kale,images


	def compute_fid(self,images):
		mu1, sigma1 = get_fid_stats(args.dataset)	
		# Be careful about the range of the images they should be from 0 to 255 !!
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mu2, sigma2 = fid_official_tf.calculate_activation_statistics(images, sess, batch_size=100)

        fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_score

def make_log_dir(args):
	if args.with_sacred:
		log_dir = args.log_dir + '_' + args.log_name
	else:
		log_dir = os.path.join(args.log_dir,args.log_name)
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	return log_dir

def get_dtype(args):
	if args.dtype=='32':
		return tf.float32
	elif args.dtype=='64':
		return tf.float64

def save_pickle(out,exp_dir,name):
	os.makedirs(exp_dir, exist_ok=True)
	with  open(os.path.join(exp_dir,name+".pickle"),"wb") as pickle_out:
		pickle.dump(out, pickle_out)

def save_csv(out, exp_dir, name,iteration):
	csv_columns = list(out.keys())
	if iteration==0:
		with open(os.path.join(exp_dir,name+".csv"),"w") as f:
			writer = csv.DictWriter(f, fieldnames=csv_columns)
			writer.writeheader()
			writer.writerow(out)
	else:
		with open(os.path.join(exp_dir,name+".csv"),"a") as f:
			writer = csv.DictWriter(f, fieldnames=csv_columns)
			writer.writerow(out)		
