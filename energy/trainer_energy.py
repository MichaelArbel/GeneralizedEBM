import math

#import tensorflow as tf
import torch


import numpy as np

# Plotting library.
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#import seaborn as sns

import numpy as np

import csv
import os
import time
import numpy as np
import pprint
import socket

# Don't forget to select GPU runtime environment in Runtime -> Change runtime type

#import tensorflow as tf
#from keras.applications.inception_v3 import InceptionV3
import torch



from helpers import *
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import dataloader



class Trainer(object):
	def __init__(self,args):
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)
		self.args = args
		self.device = assign_device(args.device)    
		self.log_dir,self.checkpoint_dir,self.samples_dir = make_log_dir(args)
		self.args.log_dir= self.log_dir
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
		self.train_loader, self.test_loader, self.valid_loader = get_data_loader(self.args)
		if self.args.problem=='conditional':
			dims = np.array([self.train_loader.dataset.y.shape[1],self.train_loader.dataset.X.shape[1]])
		else:
			dims = np.array([dataloader.dataset.X.shape[1]])

		self.discriminator = get_discriminator(self.args, np.sum(dims), self.device)
		trainable_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
		self.generator = get_generator(self.args, dims, self.device)
		
		if self.args.criterion=='kale':
			self.log_partition = nn.Parameter(torch.tensor(0.).to(self.device), requires_grad=True)
			trainable_params.append(self.log_partition)
		else:
			self.log_partition = 0.
		self.noise_gen = get_latent(self.args,dims[0], self.device)
		self.optim_d = get_optimizer(self.args,trainable_params)
		self.optim_g = get_optimizer(self.args,self.generator.parameters())
		self.scheduler_d = get_scheduler(self.args, self.optim_d)
		self.scheduler_g = get_scheduler(self.args, self.optim_g)
		self.loss = get_loss(self.args)
		
		self.fixed_z = Variable(self.noise_gen.sample([self.args.b_size]))
		self.counter =0
		self.g_loss = torch.tensor(0.)
		self.d_loss = torch.tensor(0.)
		self.const_factor = 0.01
		#path  = get_fid_stats_pytorch(self.args.dataset)
		#block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

		#self.fid_model = InceptionV3([block_idx]).to(self.device)

	def _gradient_penalty(self, real_data, generated_data):
		batch_size = real_data.size()[0]
		size_inter = min(batch_size,generated_data.size()[0])
		# Calculate interpolation
		alpha = torch.rand(size_inter, 1)
		alpha = alpha.expand_as(real_data)
		alpha = alpha.to(self.device)
		
		interpolated = alpha*real_data.data[:size_inter] + (1-alpha)*generated_data.data[:size_inter]
		#interpolated = torch.cat([real_data.data,generated_data.data],dim=0)
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


	def penalty_d(self,true_data,fake_data):
		penalty = 0.
		len_params = 0.
		if self.args.penalty_type=='l2':
			for params in self.discriminator.parameters():
				penalty += torch.sum(params**2)
		elif self.args.penalty_type=='gradient':
			penalty = self._gradient_penalty(true_data,fake_data)
		elif self.args.penalty_type=='gradient_l2':
			for params in self.discriminator.parameters():
				penalty += torch.sum(params**2)
				len_params += np.sum(np.array(list(params.shape)))
			penalty = penalty/len_params
			g_penalty = self._gradient_penalty(true_data,fake_data)
			#ratio = penalty/(g_penalty+1e-6)
			#ratio = ratio.detach()
			penalty += g_penalty
		return penalty

	def iteration(self,data,y=None,loss_type='discriminator',train_mode='train'):
		Z_factor = self.args.Z_factor
		if train_mode=='train':
			self.optim_d.zero_grad()
			self.optim_g.zero_grad()
		Z = self.noise_gen.sample([data.shape[0]*Z_factor])
		if y is None:
			gen_data_in = self.generator(Z)
			true_data_in = data
		else:
			tiled_data = data.repeat(Z_factor,1)
			gen_data_in = self.generator(Z,tiled_data)
			gen_data_in = torch.cat([tiled_data,gen_data_in],dim=1)
			true_data_in = torch.cat([data,y],dim=1)

		true_data = self.discriminator(true_data_in)
		fake_data = self.discriminator(gen_data_in)
		if self.args.criterion=='kale':
			true_data += self.const_factor*self.log_partition
			fake_data += self.const_factor*self.log_partition
		loss = self.loss(true_data,fake_data,loss_type) 
		if loss_type=='discriminator':
			loss+= self.args.penalty_lambda*self.penalty_d(true_data_in,gen_data_in)			
			optimizer = self.optim_d
		elif loss_type=='generator':
			optimizer = self.optim_g
		if train_mode=='train':
			loss.backward()
			optimizer.step()
		return loss

	def train_epoch(self,epoch):
		if self.counter==0:
			n_iter_d = self.args.n_iter_d_init
		else:
			n_iter_d = self.args.n_iter_d
		for batch_idx, (X, y) in enumerate(self.train_loader):
			X,y = X.to(self.device), y.to(self.device)
			if self.args.problem=='unconditional':
				y=None
			self.counter += 1
			if np.mod(self.counter, n_iter_d)==0:
				self.g_loss = self.iteration(X,y=y,loss_type='generator')
			else:
				self.d_loss = self.iteration(X,y=y,loss_type='discriminator')
			if batch_idx % 100 == 0:
				self.generator.Sigma = None
				if self.args.problem=='unconditional':
					neg_log_density = - self.generator.log_density(X)
				else:
					neg_log_density = - self.generator.log_density(y,X)
				neg_log_likelihood = neg_log_density.mean() + self.d_loss
				print(' Neg likelihoood: ' +  str(neg_log_likelihood.item()))

	def train_discriminator(self,epoch):
		for batch_idx, (X,y) in enumerate(self.train_loader):
			X,y = X.to(self.device), y.to(self.device)
			if self.args.problem=='unconditional':
				y=None
			self.d_loss = self.iteration(X,y=y,loss_type='discriminator')
			if batch_idx % 100 == 0:
				self.generator.Sigma = None
				if self.args.problem=='unconditional':
					neg_log_density = - self.generator.log_density(X).mean()
				else:
					neg_log_density = - self.generator.log_density(y,X).mean()
				neg_log_likelihood = neg_log_density.mean() + self.d_loss
				print(' Neg likelihoood: d_loss: ' +  str(neg_log_likelihood.item()))

	def train(self):
		# for epoch in range(self.args.total_epochs):
		# 	print('Training the base...')
		# 	print('Epoch: ' + str(epoch))
		# 	self.train_epoch(epoch)
		# 	if np.mod(epoch,10)==0:
		# 		out = self.evaluate(epoch)
		# 	#self.sample_images(epoch)
		# self.save_checkpoint(epoch)
		print('Done training the base, learning MLE')
		for epoch in range(self.args.total_epochs):
			print('Epoch: ' + str(epoch))
			self.train_discriminator(epoch)
			if np.mod(epoch,10)==0:
				out = self.evaluate(epoch)
			#self.evaluate(epoch)
		self.save_checkpoint(epoch)
		out = self.evaluate(epoch)
		return out

	def load_generator(self):
		g_model = torch.load(self.args.g_path +'.pth')
		self.noise_gen = get_latent(self.args,self.device)
		self.generator.load_state_dict(g_model)
		self.generator = self.generator.to(self.device)
	
	def load_discriminator(self):
		d_model = torch.load(self.args.d_path +'.pth')
		self.discriminator.load_state_dict(d_model)
		self.discriminator = self.discriminator.to(self.device)	

	def evaluate(self,epoch):
		neg_kale, neg_kale_std = self.acc_stats(self.train_loader)
		neg_kale_test, neg_kale_std_test = self.acc_stats(self.test_loader)
		neg_kale_valid, neg_kale_std_valid = self.acc_stats(self.valid_loader)
		print('Negative log-likelihood: train ' +  str(neg_kale) + 'Negative log-likelihood: std '+str(neg_kale_std) )
		print('Negative log-likelihood: test ' +  str(neg_kale_test) + 'Negative log-likelihood: std '+str(neg_kale_std_test) )
		print('Negative log-likelihood: valid ' +  str(neg_kale_valid) + 'Negative log-likelihood: std '+str(neg_kale_std_valid) )

		out = {}
		out['neg_kale_test'] = neg_kale_test
		out['neg_kale_std_test'] = neg_kale_std_test
		out['neg_kale'] = neg_kale_test
		out['neg_kale_std'] = neg_kale_std
		out['neg_kale_valid'] = neg_kale_valid
		out['neg_kale_std_valid'] = neg_kale_std_valid
		return out
	# def eval_pre_trained(self):
	# 	self.load_generator()
	# 	self.load_discriminator()
	# 	for epoch in range(self.args.total_epochs):
	# 		print('Epoch: ' + str(epoch))
	# 		#self.train_discriminator(epoch)
	# 		self.evaluate(epoch)
	# 		self.sample_images(epoch)
	# 		self.save_checkpoint(epoch)
	# 	self.sample_images(0)
	# 	self.evaluate(0)

	def acc_stats(self,data_loader):
		#n_batches = int(self.args.fid_samples/self.args.b_size)+1
		Z_factor = self.args.Z_factor
		n_rep = 1
		neg_log_likelihoods = []
		for _ in range(n_rep):
			mean_gen = 0.
			mean_data = 0.
			num_true = 0
			num_fake = 0
			with torch.no_grad():
				for batch_idx, (X, y) in enumerate(data_loader):
					X,y = X.to(self.device), y.to(self.device)
					Z = self.noise_gen.sample([X.shape[0]*Z_factor])
					self.generator.Sigma = None
					if self.args.problem=='unconditional':
						fake_data = self.generator(Z)
						true_data = X
						neg_log_density = - self.generator.log_density(X).sum()
					else:
						tiled_X = X.repeat(Z_factor,1)
						fake_data = self.generator(Z,tiled_X)
						true_data = torch.cat([X,y],dim=1)
						fake_data = torch.cat([tiled_X,fake_data],dim=1)
						neg_log_density = - self.generator.log_density(y,X).sum()

					true_data = self.discriminator(true_data)
					fake_data = self.discriminator(fake_data)

					true_data += self.const_factor* self.log_partition
					fake_data += self.const_factor*self.log_partition
					num_true += true_data.size(0)
					num_fake += fake_data.size(0)
	 

					fake_data = torch.exp(-fake_data).sum() 
					true_data = true_data.sum()
					
					neg_log_likelihood = (neg_log_density + true_data)/num_true +fake_data/num_fake
				neg_log_likelihood = neg_log_likelihood - 1
				neg_log_likelihoods.append(neg_log_likelihood.item())
		neg_log_likelihoods = np.array(neg_log_likelihoods)

		return np.mean(neg_log_likelihoods), np.std(neg_log_likelihoods)

	def save_checkpoint(self,epoch):
			torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, 'disc_{}'.format(epoch) ))
			torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, 'gen_{}'.format(epoch) ))
			if self.args.criterion=='kale':
				torch.save({'log_partition':self.log_partition}, os.path.join(self.checkpoint_dir, 'log_partition_{}'.format(epoch) ))

def make_log_dir(args):
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
	return log_dir,checkpoint_dir,samples_dir

def assign_device(device):
	if device >-1:
		device = 'cuda:'+str(device) if torch.cuda.is_available() and device>-1 else 'cpu'
	elif device==-1:
		device = 'cuda'
	elif device==-2:
		device = 'cpu'
	return device

def compute_fid(mu1, sigma1, mu2, sigma2):
	# calculate activations
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid





