from __future__ import print_function

#import torch
import argparse
import yaml
import torch

from trainer import Trainer, TrainerEBM

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
parser.add_argument('--log_name', type=str,   help='name for the run')
parser.add_argument('--d_path', default=None, help='path to discriminator checkpoint')
parser.add_argument('--g_path', default=None, help='path to generator checkpoint')
parser.add_argument('--log_to_file', action='store_true',  help='log stdout/stderr to logfile')
parser.add_argument('--save_nothing', action='store_true', help="don't save any files except maybe the logfile")

# control parameters
parser.add_argument('--mode', type=str, default='train',   help='train, eval_pre_trained, etc')
parser.add_argument('--train_mode', default='both', type=str,  help='if mode == train, which models to train')
parser.add_argument('--dataset', default='cifar10',type= str,  help='dataset')

# operational parameters
parser.add_argument('--device', default = 0 ,type= int,   help='gpu device')
parser.add_argument('--seed', default = 0 ,type= int ,    help='gpu device')
parser.add_argument('--dtype', default='32', type= str,   help='gpu device')
parser.add_argument('--num_workers', default=4, type=int, help='gpu device')
parser.add_argument('--total_epochs', default=100, type=int, help='total number of epochs')

# choose which generator/discriminator models to use
parser.add_argument('--g_model', default = 'dcgan' ,type= str,  help='check models/generator.py')
parser.add_argument('--d_model', default = 'vanilla' ,type= str,  help='check models/disciminator.py')
parser.add_argument('--generator', default = 'convolutional' ,type= str,  help='check models/disciminator.py')
parser.add_argument('--discriminator', default = 'convolutional' ,type= str,  help='check models/disciminator.py')

# sampling noise parameters
parser.add_argument('--lmc_gamma', default=1e-2, type=float, help='LMC parameter: gamma')
parser.add_argument('--lmc_kappa', default=4e-2, type=float, help='LMC parameter: kappa')
parser.add_argument('--num_lmc_steps', default=100, type=int, help='how many steps of LMC to run')

parser.add_argument('--slurm_id', default = '',type= str,  help='log directory for summaries and checkpoints')

# choose loss function, optimizer parameters
parser.add_argument('--criterion', default='kale',type= str, help='loss type')
parser.add_argument('--optimizer', default='Adam', type= str, help='optimizer')
parser.add_argument('--b_size', default=128, type= int,  help='default batch size')
parser.add_argument('--sample_b_size', default=1000, type= int,  help='default batch size')

parser.add_argument('--fid_b_size', default=128, type= int,  help='default batch size')

parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--sgd_momentum', default=0., type=float, help='learning rate')
parser.add_argument('--beta_1', default=0.5, type=float, help='learning rate')
parser.add_argument('--beta_2', default=0.9, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0, type=float, help='learning rate')

# choose learning rate of the generator, if training it
parser.add_argument('--lr_generator', default=0.0002, type=float, help='lr')

# regularization
parser.add_argument('--penalty_type', default = 'gradient_l2',type= str,  help='type of regularization to add')
parser.add_argument('--penalty_lambda', default=0.1, type=float, help='learning rate')

# training parameters

parser.add_argument('--n_iter_d_init', default = 100, type= int,  help='number of initial discriminator updates')
parser.add_argument('--n_iter_d', default = 5, type= int,  help='number of discriminator update p/generator update')
parser.add_argument('--Z_dim', default = 128, type= int,  help='dimension of latent')

# Scheduler parameters 
parser.add_argument('--use_scheduler', action='store_true', help='gpu device')
parser.add_argument('--scheduler',  default ='MultiStepLR' ,type= str ,  help='gpu device')
parser.add_argument('--milestone',  default = '10,50,70' ,type= str ,  help='gpu device')
parser.add_argument('--scheduler_gamma',  default=0.8, type= float, help='gpu device')
parser.add_argument('--lr_decay',  default = 0.8 ,type= float ,  help='learning rate decay')


# others
parser.add_argument('--configs',  default ='' ,type= str ,  help='use config file')

parser.add_argument('--eval_fid', action='store_true', help='calculate FID, but takes time')
parser.add_argument('--eval_kale', action='store_true', help='calculate FID, but takes time')

parser.add_argument('--fid_samples', default = 50000, type= int,  help='number of samples to take to calculate FID')
parser.add_argument('--lmc_sampler_chain', action='store_true', help='calculate FID, but takes time')

parser.add_argument('--latent_sampler',  default ='langevin' ,type= str ,  help='calculate FID, but takes time')

parser.add_argument('--num_sampler_steps', default = 1000, type= int,  help='number of samples to take to calculate FID')


parser.add_argument('--bn', action='store_true', help='calculate FID, but takes time')
parser.add_argument('--skipinit', action='store_true', help='calculate FID, but takes time')
parser.add_argument('--train_mode_fid', action='store_true', help='calculate FID, but takes time')

parser.add_argument('--dataparallel', action='store_true', help='calculate FID, but takes time')

parser.add_argument('--imagenet_train_path',  default ='' ,type= str ,  help='calculate FID, but takes time')
parser.add_argument('--imagenet_test_path',  default ='' ,type= str ,  help='calculate FID, but takes time')
parser.add_argument('--truncation',  default = 1. ,type= float ,  help='learning rate decay')
parser.add_argument('--trunc',  default = 2. ,type= float ,  help='learning rate decay')
parser.add_argument('--optimal_log_partition', action='store_true', help='calculate FID, but takes time')
parser.add_argument('--lr_partition',  default = 1. ,type= float ,  help='learning rate decay')

parser.add_argument('--total_gen_iter',  default = 150000, type= float ,  help='learning rate decay')
parser.add_argument('--test', default = 0, type= int,  help='number of samples to take to calculate FID')
parser.add_argument('--initialize_log_partition', action='store_true', help='calculate FID, but takes time')

parser.add_argument('--temperature', default = 100. ,type= float , help='calculate FID, but takes time')
parser.add_argument('--noise_factor', default = 1, type= int,  help='number of samples to take to calculate FID')
parser.add_argument('--trainer_type', default ='default' ,type= str , help='number of samples to take to calculate FID')
parser.add_argument('--dataset_type', default ='images' ,type= str , help='[images, uci]')
parser.add_argument('--latent_noise', default ='gaussian' ,type= str , help='[images, uci]')
parser.add_argument('--freq_fid', default = 2000, type= int, help='[images, uci]')
parser.add_argument('--freq_kale', default = 2000, type= int , help='[images, uci]')
parser.add_argument('--disp_freq', default = 100, type= int , help='[images, uci]')
parser.add_argument('--checkpoint_freq', default = 1000, type= int , help='[images, uci]')


args = parser.parse_args()
args = make_flags(args, args.configs)
if args.trainer_type=='default':
	trainer = Trainer(args)
elif args.trainer_type=='ebm':
	trainer = TrainerEBM(args)
# check whether we want to load a pretrained model depending on the given parameters


trainer.main()
print('Finished!')







