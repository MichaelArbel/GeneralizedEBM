import os
import time

import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# fid_pytorch, inception
import metrics.fid_pytorch as fid_pytorch
from metrics.inception import InceptionV3
import torch
from scipy import linalg 

def wasserstein(true_data,fake_data,loss_type):
    if loss_type=='discriminator':
        return -true_data.mean() + fake_data.mean()
    else:
        return -fake_data.mean()
def logistic(true_data,fake_data,loss_type):
    if loss_type =='discriminator':
        loss = torch.nn.BCEWithLogitsLoss()(true_data, torch.ones(true_data.shape[0]).to(true_data.device)) + \
                    torch.nn.BCEWithLogitsLoss()(fake_data, torch.zeros(fake_data.shape[0]).to(fake_data.device))
        return loss
    else:
        loss = torch.nn.BCEWithLogitsLoss()(fake_data, torch.ones(fake_data.shape[0]).to(fake_data.device))
        return loss
def kale(true_data,fake_data,loss_type):
    if loss_type=='discriminator':
        return  true_data.mean() + torch.exp(-fake_data).mean()  - 1
    else:
        return -true_data.mean() #- torch.exp(-fake_data).mean()  + 1


# calculates regularization penalty term for learning
def penalty_d(args, d, true_data, fake_data, device):
    penalty = 0.
    len_params = 0.
    # no penalty
    if args.penalty_type == 'none':
        pass
    # L2 regularization only
    elif args.penalty_type=='l2':
        for params in d.parameters():
            penalty += torch.sum(params**2)
    # gradient penalty only
    elif args.penalty_type=='gradient':
        penalty = _gradient_penalty(d, true_data, fake_data, device)
    # L2 + gradient penalty
    elif args.penalty_type=='gradient_l2':
        for params in d.parameters():
            penalty += torch.sum(params**2)
            len_params += np.sum(np.array(list(params.shape)))
        penalty = penalty/len_params
        g_penalty = _gradient_penalty(d, true_data, fake_data, device)
        penalty += g_penalty
    return penalty

# helper function to calculate gradient penalty
# adapted from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
def _gradient_penalty(d, true_data, fake_data, device):
    batch_size = true_data.size()[0]
    size_inter = min(batch_size,fake_data.size()[0])
    # Calculate interpolation
    shape  = list(np.ones(len(true_data.shape)-1))
    shape = tuple([int(a) for a in shape])
    alpha = torch.rand((size_inter,)+shape)
    alpha = alpha.expand_as(true_data)
    alpha = alpha.to(device)
    
    interpolated = alpha*true_data.data[:size_inter] + (1-alpha)*fake_data.data[:size_inter]
    #interpolated = torch.cat([true_data.data,fake_data.data],dim=0)
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = d(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sum(gradients ** 2, dim=1).mean()
    return gradients_norm




def iterative_mean(batch_tensor, total_mean, total_els, dim=0 ):
    b = batch_tensor.shape[dim]
    cur_mean = batch_tensor.mean(dim=dim)
    total_mean = (total_els/(total_els+b))*total_mean + (b/(total_els+b))*cur_mean
    total_els += b
    return total_mean, total_els

def iterative_log_sum_exp(batch_tensor,total_sum,total_els, dim=0):
    b = batch_tensor.shape[dim]
    cur_sum = torch.logsumexp(batch_tensor, dim=0).sum()
    total_sum = torch.logsumexp(torch.stack( [total_sum,cur_sum] , dim=0), dim=0).sum()
    total_els += b  
    return total_sum,  total_els


def compute_nll(data_loader, model, device):
    model.eval()
    log_density = 0.
    M = 0
    for i, (data,target) in enumerate(data_loader): 
        with torch.no_grad():
            cur_log_density = - model.log_density(data.to(device)) 
            log_density, M = iterative_mean(cur_log_density, log_density,M)

    return log_density.mean()




def get_fid_stats(fid_model, loader, dataset, dataset_type, device):
    path = 'metrics/res/stats_pytorch/fid_stats_'+dataset+'_'+dataset_type+'.npz'
    try:
        f = np.load(path)
        mu, sigma = f['mu'][:], f['sigma'][:]
        f.close()    
    except:
        print('==> Computing data stats')
        mu, sigma = fid_pytorch.compute_stats_from_loader(fid_model,loader, device)
        np.savez(path, mu=mu, sigma=sigma)
    return mu, sigma

def get_activations_from_loader(dataloader, model, device,total_samples=50000,batch_size=50, dims=2048, verbose=False, is_tuple=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #n_batches = len(dataloader.dataset.data) // batch_size
    #n_used_imgs = n_batches * batch_size

    pred_arr = []
    num_samples = 0
    for batch_idx, data in enumerate(dataloader):
        if is_tuple:
            data,_ = data
        if num_samples<=total_samples:
            if verbose:
                print('\rPropagating batch %d' % (batch_idx + 1),
                      end='', flush=True)
            start = batch_idx * batch_size
            end = start + batch_size
            image_batch = (data+1.)*0.5
            batch = image_batch.to(device)
            with torch.no_grad():
                pred = model(batch)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.mean([2,3])
                pred_arr.append(pred.cpu())
            num_samples += data.shape[0]
        else:
            break
    pred_arr = torch.cat(pred_arr, dim=0)
    if verbose:
        print(' done')
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # taken from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
