## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
   * [Training](#cifar10)
   * [Sampling](#cifar100)
* [Resources](#resources)
   * [Data](#data)
   * [Hardware](#hardware)
* [Full documentation](#full-documentation)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the training and sampling methods proposed in [Generalized Energy Based Models](https://arxiv.org/abs/2003.05033) along with scripts to reproduce results of the paper.

Samples using Underdamped Langevin dynamics:

![Cifar10](lmc_cifar.gif)
![ImageNet32](lmc_imagenet.gif)

Samples using Overdamped Langevin dynamics:

![Cifar10](langevin_cifar.gif)
![ImageNet32](langevin_imagenet.gif)

## Requirements


This a Pytorch implementation which requires the follwoing packages:

```
python==3.6.2 or newer
torch==1.4.0 or newer
torchvision==0.5.0 or newer
numpy==1.17.2  or newer
```

All dependencies can be installed using:

```
pip install -r requirements.txt
```




## How to use


### Training
```
python main.py --config=configs/training.yaml --dataset=cifar10
```

### Sampling

```
python main.py --config=configs/sampling.yaml --dataset=cifar10 --latent_sampler=langevin --lmc_gamma=0.0001
```




## Resources

### Data

To be able to reproduce the results of the paper using the prodivided scripts, the datasets need to be downloaded. This is automatically done by the script for Cifar10. By default a directory named 'data' containing the datasets is created in the working directory. 


### Hardware

To use a particular GPU, set 
```'—device=#gpu_id'```

To use GPU without specifying a particular one, set 
```—device=-1```

To use CPU set 
```—device=-2```


## Full documentation

```
--log_name                  log name ['']
--log_dir                   log directory for summaries and checkpoints ['']
--d_path                    path to the trained energy network
--g_path                    path to the trained base network
--data_path                 directory to the dataset ['data']
--imagenet_train_path       path to imagenet train set
--imagenet_test_path        path to imagenet test set


--log_in_file               log output in a file [False]
--save_nothing              Disable saving of the model [False]
--disp_freq                 frequency for displaying the loss [100]
--checkpoint_freq           frequency for saving checkpoints [1000]

--mode                      either 'train' or 'sample' 
--train_mode                either train both energy and base or only one of them ['both','energy','base']
--dataset                   name of the dataset to use  ['cifar10','CelebA','Imagenet32','lsun']


--device                    gpu device [0]
--seed                      seed for randomness [0]
--dtype                     32 for float32 and 64 for float64 ['32']
--num_workers               Number of workers of the dataloader ['4']
--dataparallel              parallelize over multiple gpus [False]
--slurm_id                  job id when using slurm, modified internally ['']
--trainer_type              the class for training / sampling ['default']
--dataset_type              image dataset or others ['images','uci']


# Model parameters
--g_model                   architecture of the base network: ['dcgan','sngan']
--d_model                   architecture of the energy network: ['dcgan','sngan']
--generator                 network type of the base : ['convolutional']
--discriminator             network type of the energy : ['convolutional']
--latent_noise              the distribution of latent noise ['gaussian']
--bn                        batch-normalization [False]
--num_blocks                number of blocks for the NVP [3]

# Sampling parameters
--latent_sampler            which sampler to use ['langevin','mala','lmc','hmc']
--lmc_gamma                 step-size for the lmc sampler: [1e-2]
--lmc_kappa                 friction parameter of the lmc sampler: ['100.']
--num_sampler_steps         number of sampler steps [100]
--temperature               temperature parameter [100]

# Batch size
--fid_b_size                batch-size for computing FID [128]
--sample_b_size             batch-size for sampling [1000]
--b_size                    batch_size for training [128]

# criterion
--criterion                 top level loss ['kale']
--penalty_type              the penalty for training the energy ['gradient_l2','gradient','l2','none']
--penalty_lambda            strenght of the penalty [.1]
--initialize_log_partition  initialize log-partition using Monte-Carlo estimator [False]

--total_gen_iter            total number of iterations for the base [150000]
--total_epochs              total number of epochs [100]
--n_iter_d_init             number of iterations on the energy at the begining of training and every 500 iterations on the base [100]
--n_iter_d                  number of iterations on the energy for every training iteration on the base [5]
--noise_factor              factor multiplying the data batch size and giving the latent samples batch size [1]



# Optimizer parameters
--optimizer                 Inner optimizer to compute the euclidean gradient['Adam']
--lr                        learning rate for the energy[.00001]
--lr_generator              learning rate for the base [.0002]
--sgd_momentum              momentum parameter for SGD [0.]
--beta_1                    first parameter of Adam optimizer [.5]
--beta_2                    second parameter of Adam optimizer [.9]
--weight_decay              weight decay [0.]

# Scheduler parameters 
--use_scheduler             schedule the lr ['store_true']
--scheduler                 scheduler ['MultiStepLR']
--milestone                 decrease schedule for lr at epochs  ['10,50,70']
--scheduler_gamma           decay of the learning rate ['.8']
--lr_decay                  decay of the learning rate ['.8']

# Metrics
--eval_kale                 evaluate KALE on both training and test sets ['False']
--fres_kale                 frequency for evaluating kale per iteratations [2000]
--eval_fid                  evaluate the FID scores [False]
--fid_samples               number of generated samples to evaluate the score [50000]
--freq_fid                  frequency for evaluating FID per iteratations [2000]

# Config path
--configs                   config file for the run ['']
```

## Reference

If using this code for research purposes, please cite:

[1] M. Arbel, L. Zhou and A. Gretton [*Generalized Energy Based Models*](https://arxiv.org/abs/2003.05033)

```
@article{arbel2020kale,
  title={Generalized Energy Based Models},
  author={Arbel, Michael and Zhou, Liang and Gretton, Arthur},
  journal={arXiv preprint arXiv:2003.05033},
  year={2020}
}
```


## License 

This code is under a BSD license.
