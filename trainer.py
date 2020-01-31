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



class Trainer(object):
	def __init__(self,args):
		self.args = args
		#self.device = assign_device(args.device)	
		self.log_dir = make_log_dir(args)
		if args.log_in_file:
			self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
			sys.stdout = self.log_file
			sys.stderr = self.log_file
		print("Process id: " + str(os.getpid()) + " | hostname: " + socket.gethostname())
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(args))

		device_name = tf.test.gpu_device_name()
		if device_name != '/device:GPU:0':
			raise SystemError('GPU device not found')
		print('Found GPU at: {}'.format(device_name))

		print('==> Building model..')
		self.build_model()


	def build_model(self):
		tf.random.set_seed(self.args.seed)
		self.model = get_model(self.args)
		self.dist1, self.dist2 = get_dist(self.args)
		self.transform = get_transform(self.args)

		self.opt = get_optimizer(self.args)
		self.eval_model = None
		self.counter = 0
		#self.scheduler = get_scheduler(self.args,self.optimizer)

	def estimate_kl(self,sample_size):
		d1 = self.model(self.transform(self.dist1.sample(sample_size)))
		d2 = tf.exp(self.model(self.transform(self.dist2.sample(sample_size))))
		return tf.reduce_mean(1 + d1 - d2)
	def estimate_eval_kl(self,sample_size):

		d1 = self.eval_model(self.transform(self.dist1.sample(sample_size)))
		d2 = tf.exp(self.eval_model(self.transform(self.dist2.sample(sample_size))))
		return tf.reduce_mean(1 + d1 - d2)	

	def step(self):
		with tf.GradientTape() as tape:
			estimated_kl = self.estimate_kl((self.args.b_size, 1))
			estimated_kl = tf.reduce_mean(estimated_kl)
			loss = - estimated_kl
			params = self.model.trainable_variables
			grads = tape.gradient(loss, params)
			self.opt.apply(grads, params)

	def train(self):

		#print(' Starting training')
		done = False
		while not done:
			for i in range(self.args.total_iter):
				self.step()
				self.update_eval_model(i)
				if i % 100 == 0:
					print(self.estimate_eval_kl((1000, 1)))
			#print(' Done training')

			est = self.estimate_eval_kl((100000, 1)).numpy()
			if est >0:
				done = True
		
		#out = self.eval()
		#save_pickle(out,os.path.join(self.log.dir), name =  'iter_'+ str(args.total_iter).zfill(8))

		return est

	def update_eval_model(self,iteration):
		if iteration<=self.args.iteration_min:
			self.eval_model = deepcopy(self.model)
			self.counter = 1
		elif iteration>self.args.iteration_min:
			params = list(self.model.trainable_variables)
			eval_params = list(self.eval_model.trainable_variables )
			counter = self.counter
			runing_avg = [(counter* eval_param + param)/(counter+1) for eval_param,param in zip(eval_params,params)]
			for update, param in zip(runing_avg,self.eval_model.trainable_variables):
				param.assign(update)
			self.counter += 1


	def eval(self):
		out ={}
		out['mean_kale'],out['std_kale'],out['mean_kale_avg'],out['std_kale_avg'] = self.eval_kl()
		#out['mean_kale'],out['std_kale'] = self.eval_kl()
		out['total_kl_eval'] = self.args.total_kl_eval
		out['num_sample_eval'] = self.args.num_sample_eval
		out['KL'] 	= self.dist1.kl_divergence(self.dist2).numpy()
		out['center_1'] = self.dist1.loc.numpy()
		out['center_2'] = self.dist2.loc.numpy()
		out['sigma_1'] = self.dist1.scale.numpy()
		out['sigma_2'] = self.dist2.scale.numpy()		#if self.args.model =='synthetic':
		out['smoothness'] = self.args.smoothness
		print('kale: '+ str(out['mean_kale']) + ', kale_avg: ' +str(out['mean_kale_avg']) + ', KL: '+str(out['KL']) )
		return out

	def params_loop(self):
		all_center_2 = np.linspace(self.args.center_1+self.args.center_offset_min,self.args.center_1+self.args.center_offset_max,self.args.num_params_vals)
		all_smoothness = np.logspace(self.args.smoothness_max,self.args.smoothness_min,self.args.num_params_vals)
		
		#all_smoothness = np.array([100,100.,10.,10,1.,1.,0.001,0.001])
		#all_center_2 = np.array([1.,2.,3.,3.2,3.4,3.6,3.8,4.,5.])

		k = 0
		for center_2, smoothness in zip(all_center_2,all_smoothness):
			self.args.smoothness = smoothness
			self.args.center_2 = center_2
			self.build_model()
			est = self.train()
			out = self.eval()
			save_csv(out, os.path.join(self.log_dir), 'all_out',k)
			print('Iteration: '+ str(k) + 'final estimated value: '+str(est) )
			k +=1

	def eval_kl(self):
		kl_eval = np.zeros([self.args.total_kl_eval])
		kl_eval_avg = np.zeros([self.args.total_kl_eval])
		for k in range(self.args.total_kl_eval):
			val = self.estimate_kl((self.args.num_sample_eval, 1))
			val_avg = self.estimate_eval_kl((self.args.num_sample_eval, 1))
			kl_eval[k] = val.numpy()
			kl_eval_avg[k] = val_avg.numpy()
		return np.mean(kl_eval[kl_eval>0.]), np.std(kl_eval[kl_eval>0.]),np.mean(kl_eval_avg[kl_eval_avg>0.]),np.std(kl_eval_avg[kl_eval_avg>0.])

# def assign_device(device):
# 	if device >-1:
# 		device = 'cuda:'+str(device) if torch.cuda.is_available() and device>-1 else 'cpu'
# 	elif device==-1:
# 		device = 'cuda'
# 	elif device==-2:
# 		device = 'cpu'
# 	return device
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


def get_model(args):
	if args.model=='simple':

		model = snt.Sequential([
			snt.Linear(200),
			tf.nn.leaky_relu,
			snt.Linear(200),
			tf.nn.leaky_relu,
			snt.Linear(1),])
	else:
		raise NotImplementedError()

	return model


def get_dist(args):
	
	dist1 = tfd.Normal(float(args.center_1),float( args.sigma_1))
	dist2 = tfd.Normal(float(args.center_2), float(args.sigma_2))
	return dist1, dist2

def get_optimizer(args):

	if args.optimizer=='sgd':
		return snt.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)
	elif args.optimizer=='adam':
		return snt.optimizers.Adam(learning_rate=args.lr,beta1=args.beta_1,beta2=args.beta_2)
	elif args.optimizer=='rmsprop':
		return snt.optimizers.RMSProp(learning_rate=args.lr,decay=args.lr_decay,momentum=args.momentum)
def get_transform(args):
	if args.transform=='monotonic_fourier':
		def transform(x):
			f = lambda x: utils.monotonic_fourier(x, args.L,args.J,args.off_set,args.smoothness,args.alpha,args.seed_monotonic)
			if not isinstance(x, Iterable):
				x = np.ones((1, 1)) * x
				return f(x)[0, 0]
			if x.ndim == 1:
				x = np.expand_dims(x, axis=1)
				return f(x)[:, 0]
			return f(x)
		return transform
	elif args.transform=='id_transform':
		return utils.id_transform

# def get_scheduler(args,optimizer):
# 	if args.scheduler=='MultiStepLR':
# 		if args.milestone is None:
# 			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.total_epochs*0.5), int(args.total_epochs*0.75)], gamma=args.lr_decay)
# 		else:
# 			milestone = [int(_) for _ in args.milestone.split(',')]
# 			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=args.lr_decay)
# 		return lr_scheduler

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
