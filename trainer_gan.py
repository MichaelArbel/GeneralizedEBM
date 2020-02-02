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
from metrics.inception import InceptionV3
import torch



from helpers import *
from torch.autograd import Variable

import metrics.fid_pytorch as fid_pytorch




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
		self.train_loader, self.test_loader = get_data_loader(self.args)
		

		self.discriminator = get_net(self.args, 'discriminator', self.device)
		trainable_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
		self.generator = get_net(self.args, 'generator', self.device)
		
		if self.args.criterion=='kale':
			self.log_partition = nn.Parameter(torch.tensor(0.).to(self.device), requires_grad=True)
			trainable_params.append(self.log_partition)
		else:
			self.log_partition = 0.
		self.optim_d = get_optimizer(self.args,trainable_params)
		self.optim_g = get_optimizer(self.args,self.generator.parameters())
		self.scheduler_d = get_scheduler(self.args, self.optim_d)
		self.scheduler_g = get_scheduler(self.args, self.optim_g)
		self.loss = get_loss(self.args)
		self.noise_gen = get_latent(self.args,self.device)
		#self.fixed_z = Variable(self.noise_gen.sample([self.args.b_size]))

		#self.penalty_d = get_penatly(self.args)

		self.counter =0
		self.g_loss = torch.tensor(0.)
		self.d_loss = torch.tensor(0.)
		path  = get_fid_stats_pytorch(self.args.dataset)
		block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

		# self.fid_model = InceptionV3([block_idx]).to(self.device)

	def iteration(self,data,loss_type,train_mode='train'):
		if train_mode=='train':
			self.optim_d.zero_grad()
			self.optim_g.zero_grad()
		Z = self.noise_gen.sample([self.args.b_size])
		gen_data = self.generator(Z)
		true_data = self.discriminator(data)
		fake_data = self.discriminator(gen_data)
		if self.args.criterion=='kale':
			true_data += self.log_partition
			fake_data += self.log_partition
		loss = self.loss(true_data,fake_data,loss_type) 
		if loss_type=='discriminator':
			#loss+= self.args.penalty_lambda*self.penalty_d(true_data,fake_data)			
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
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data = Variable(data.to(self.device))
			self.counter += 1
			if np.mod(self.counter, n_iter_d)==0:
				self.g_loss = self.iteration(data,'generator')
			else:
				self.d_loss = self.iteration(data,'discriminator')
			if batch_idx % 100 == 0:
				print('generator loss: '+ str(self.g_loss.item())+', critic loss: ' +  str(self.d_loss.item()))

	def train_discriminator(self,epoch):
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data = Variable(data.to(self.device))
			self.d_loss = self.iteration(data,'discriminator')
			if batch_idx % 100 == 0:
				print(' critic loss: ' +  str(self.d_loss.item()))

	def train(self):
		for epoch in range(self.args.total_epochs):
			print('Epoch: ' + str(epoch))
			self.train_epoch(epoch)
			self.evaluate(epoch)
			self.sample_images(epoch)
			self.save_checkpoint(epoch)

	def load_generator(self):
		g_model = torch.load(self.args.g_path +'.pth')
		self.noise_gen = get_latent(self.args,self.device)
		self.generator.load_state_dict(g_model)
		self.generator = self.generator.to(self.device)
	
	def load_discriminator(self):
		d_model = torch.load(self.args.d_path +'.pth')
		self.discriminator.load_state_dict(d_model)
		self.discriminator = self.discriminator.to(self.device)	

	# samples 64 images according to all types in the the sample_type argument, saves them
	def sample_images(self, epoch):
		sample_types = self.args.sample_type.split(',')
		for s in sample_types:
			sample_z = get_latent_samples(self.args, self.device, s_type=s, g=self.generator, h=self.discriminator)
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

	def evaluate(self,epoch):
		if np.mod(epoch,10)==0:
			Kale,images = self.acc_stats()
			fid = self.compute_fid(images)
			print('Kale ' +  str(Kale.item()) + ', FID: '+ str(fid))
	def eval_pre_trained(self):
		self.load_generator()
		self.load_discriminator()
		for epoch in range(self.args.total_epochs):
			print('Epoch: ' + str(epoch))
			#self.train_discriminator(epoch)
			self.evaluate(epoch)
			self.sample_images(epoch)
			self.save_checkpoint(epoch)
		self.sample_images(0)
		# self.evaluate(0)

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
					if self.args.criterion=='kale':
						
						fake_data += self.log_partition						
					mean_gen += -torch.exp(-fake_data).sum()

					lengthstuff= min(self.args.fid_samples-m,gen_data.shape[0])
					if m==0:
						images = torch.zeros([self.args.fid_samples]+list(gen_data.shape[1:]))
					images[m:m+lengthstuff,:]=gen_data[:lengthstuff,:].detach().cpu()
					m = m + gen_data.size(0)
			mean_gen /=  m
			m = 0
			for batch_idx, (data, target) in enumerate(self.test_loader):
				data = Variable(data.to(self.device))
				true_data = self.discriminator(data)
				true_data += self.log_partition
				mean_data += -true_data.sum()
				m += true_data.size(0)
			mean_data /= m
			Kale = mean_data + mean_gen + 1

		return Kale,images

	def compute_fid(self,images):
		mu2, sigma2 = fid_pytorch.compute_stats(images, self.fid_model,self.device,batch_size=128 )
		try:
			mu1_train, sigma1_train = get_fid_stats(self.args.dataset+'_train')
		except:
			mu1_train, sigma1_train = self.compute_inception_stats('train',self.train_loader)
		fid_train = fid_pytorch.calculate_frechet_distance(mu1_train, sigma1_train, mu2, sigma2)

		
		if 	self.test_loader is not None:
			try:
				mu1_valid, sigma1_valid = get_fid_stats(self.args.dataset+'_valid')
			except:
				mu1_valid, sigma1_valid = self.compute_inception_stats('valid',self.test_loader)
			fid_valid = fid_pytorch.calculate_frechet_distance(mu1_valid, sigma1_valid, mu2, sigma2)

			return fid_train,fid_valid
			
		return fid_score

	def compute_inception_stats(self,type_dataset,data_loader):
		mu_train,sigma_train= fid_pytorch.compute_stats_from_loader(self.fid_model,data_loader,self.device)
		#mu_1,sigma_1 = get_fid_stats(self.args.dataset+'_'+type_dataset)
		path = 'metrics/res/stats_pytorch/fid_stats_'+self.args.dataset+'_'+type_dataset+'.npz'
		np.savez(path, mu=mu_train, sigma=sigma_train)
		#if self.test_loader is not None:
		#	mu_test,sigma_test= fid_pytorch.compute_stats_from_loader(self.fid_model,self.test_loader,self.device)
		return mu_train,sigma_train

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


# unused
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
