import torch
import torch.nn.functional as F
from torch.autograd import Variable

def hinge(true_data, fake_data, loss_type):
	if loss_type=='discriminator':
		return F.relu(1.0- true_data).mean()+ F.relu(1.0+ fake_data).mean()
	else:
		return -discriminator(generator(z)).mean()
def wasserstein(true_data,fake_data,loss_type):
	if loss_type=='discriminator':
		return -true_data.mean() + fake_data.mean()
	else:
		return -fake_data.mean()
def logistic(true_data,fake_data,loss_type):
	if loss_type =='discriminator':
		loss = torch.nn.BCEWithLogitsLoss()(true_data, Variable(torch.ones(true_data.shape[0], 1).to(true_data.device))) + \
					torch.nn.BCEWithLogitsLoss()(fake_data, Variable(torch.zeros(fake_data.shape[0], 1).to(fake_data.device)))
		return loss
	else:
		loss = torch.nn.BCEWithLogitsLoss()(fake_data, Variable(torch.ones(fake_data.shape[0], 1).to(fake_data.device)))
		return loss

def kale(true_data,fake_data,loss_type):
	if loss_type=='discriminator':
		return  true_data.mean() + torch.exp(-fake_data).mean()  
	else:
		return -true_data.mean() - torch.exp(-fake_data).mean()  
