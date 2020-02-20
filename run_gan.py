from __future__ import print_function

#import torch
import argparse
import yaml
import torch

from trainer_gan import Trainer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_flags(args, config_file):
    if config_file:
        config = yaml.safe_load(open(config_file))
        dic = vars(args)
        all(map(dic.pop, config))
        dic.update(config)
    return args

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# log parameters
parser.add_argument('--log_dir', default='logs', type= str, help='top-level logs directory')
parser.add_argument('--log_name', type=str, help='name for the run')
parser.add_argument('--d_path', default=None, help='path to discriminator checkpoint')
parser.add_argument('--g_path', default=None, help='path to generator checkpoint')
parser.add_argument('--log_to_file', action='store_true', help='log stdout/stderr to logfile')
parser.add_argument('--save_nothing', action='store_true', help="don't save any files except maybe the logfile")

# control parameters
parser.add_argument('--mode', type=str, default='train', help='train, eval_pre_trained, etc')
parser.add_argument('--train_which', default='discriminator', type=str, help='if mode == train, which models to train')
parser.add_argument('--dataset', default='cifar10',type= str,  help='dataset')

# operational parameters
parser.add_argument('--device', default = 0 ,type= int,  help='gpu device')
parser.add_argument('--seed', default = 0 ,type= int ,  help='gpu device')
parser.add_argument('--dtype', default='32', type= str, help='gpu device')
parser.add_argument('--num_workers', default=4, type=int, help='gpu device')
parser.add_argument('--total_epochs', default=100, type=int, help='total number of epochs')

# choose which generator/discriminator models to use
parser.add_argument('--g_model', default = 'dcgan' ,type= str,  help='check models/generator.py')
parser.add_argument('--d_model', default = 'vanilla' ,type= str,  help='check models/disciminator.py')

parser.add_argument('--Z_folder', default='', type=str, help='stored Zs')
parser.add_argument('--bb_size', type=int, default=1000, help='# Zs per batch, running out of memory is bad')

# sampling noise parameters
parser.add_argument('--lmc_gamma', default=1e-2, type=float, help='LMC parameter: gamma')
parser.add_argument('--lmc_kappa', default=4e-2, type=float, help='LMC parameter: kappa')
parser.add_argument('--num_lmc_steps', default=100, type=int, help='how many steps of LMC to run')


# choose loss function, optimizer parameters
parser.add_argument('--criterion', default='kale',type= str, help='loss type')
parser.add_argument('--optimizer', default='Adam', type= str, help='optimizer')
parser.add_argument('--b_size', default=128, type= int,  help='default batch size')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--sgd_momentum', default=0., type=float, help='learning rate')
parser.add_argument('--beta_1', default=0.9, type=float, help='learning rate')
parser.add_argument('--beta_2', default=0.999, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0, type=float, help='learning rate')

# choose learning rate of the generator, if training it
parser.add_argument('--lr_generator', default=0.0002, type=float, help='lr')

# regularization
parser.add_argument('--penalty_type', default = 'gradient_l2',type= str,  help='type of regularization to add')
parser.add_argument('--penalty_lambda', default=0.001, type=float, help='learning rate')

# training parameters

parser.add_argument('--n_iter_d_init', default = 100, type= int,  help='number of initial discriminator updates')
parser.add_argument('--n_iter_d', default = 5, type= int,  help='number of discriminator update p/generator update')
parser.add_argument('--Z_dim', default = 128, type= int,  help='dimension of latent')

# Scheduler parameters 
parser.add_argument('--use_scheduler', action='store_true', help='gpu device')
parser.add_argument('--scheduler',  default ='ExponentialLR' ,type= str ,  help='gpu device')
parser.add_argument('--milestone',  default = '100,200,300' ,type= str ,  help='gpu device')
parser.add_argument('--scheduler_gamma',  default=0.99, type= float, help='gpu device')
parser.add_argument('--lr_decay',  default = 0.9 ,type= float ,  help='learning rate decay')


# others
parser.add_argument('--config',  default ='' ,type= str ,  help='use config file')

parser.add_argument('--with_fid', action='store_true', help='calculate FID, but takes time')
parser.add_argument('--fid_samples', default = 50000, type= int,  help='number of samples to take to calculate FID')





args = parser.parse_args()
args = make_flags(args, args.config)
trainer = Trainer(args)

# check whether we want to load a pretrained model depending on the given parameters


trainer.main()
print('Finished!')







