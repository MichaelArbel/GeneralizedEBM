from __future__ import print_function

#import torch
import argparse
import yaml

from trainer import Trainer
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark=False



def make_flags(args,config_file):
    if config_file:
        config = yaml.load(open(config_file))
        dic = vars(args)
        all(map(dic.pop, config))
        dic.update(config)
    return args

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log_name', default = '',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_dir', default = '',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--dataset', default = 'cifar10',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--data_dir', default = 'data',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_in_file', action = 'store_true' ,  help='gpu device')

parser.add_argument('--device', default = 0 ,type= int,  help='gpu device')
parser.add_argument('--seed', default = 0 ,type= int ,  help='gpu device')
parser.add_argument('--dtype',  default = '32' ,type= str ,   help='gpu device')



parser.add_argument('--total_iter', default=1000, type=int, help='total number of epochs')
parser.add_argument('--total_kl_eval', default=1000, type=int, help='total number of epochs')
parser.add_argument('--num_sample_eval', default=100000, type=int, help='total number of epochs')



parser.add_argument('--model', default = 'simple' ,type= str,  help='gpu device')

# Optimizer parameters
parser.add_argument('--optimizer', default = 'adam', type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--b_size', default = 5000, type= int,  help='gpu device')
parser.add_argument('--lr', default=.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0., type=float, help='learning rate')
parser.add_argument('--beta_1', default=0.9, type=float, help='learning rate')
parser.add_argument('--beta_2', default=0.999, type=float, help='learning rate')

parser.add_argument('--lr_decay',  default = 0.9 ,type= float ,  help='gpu device')

# Scheduler parameters 
parser.add_argument('--use_scheduler', default = 'store_true' ,  help='gpu device')
parser.add_argument('--scheduler',  default ='MultiStepLR' ,type= str ,  help='gpu device')
parser.add_argument('--milestone',  default = '100,200,300' ,type= str ,  help='gpu device')


parser.add_argument('--config',  default ='' ,type= str ,  help='gpu device')
parser.add_argument('--with_sacred',  default =False ,type= bool ,  help='gpu device')



parser.add_argument('--center_1', default = 0. ,type= float ,  help='gpu device')
parser.add_argument('--center_offset_max', default = 5. ,type= float ,  help='gpu device')
parser.add_argument('--center_offset_min', default = 1. ,type= float ,  help='gpu device')

parser.add_argument('--smoothness_min', default = -3 ,type= float ,  help='gpu device')
parser.add_argument('--smoothness_max', default = 2. ,type= float ,  help='gpu device')
parser.add_argument('--sigma_1', default = 1. ,type= float ,  help='gpu device')
parser.add_argument('--sigma_2', default = 1. ,type= float ,  help='gpu device')
parser.add_argument('--center_2', default = 0. ,type= float ,  help='gpu device')
parser.add_argument('--smoothness', default = 100. ,type= float ,  help='gpu device')
parser.add_argument('--num_params_vals', default = 100 ,type= int ,  help='gpu device')


parser.add_argument('--alpha', default = 1. ,type= float ,  help='gpu device')
parser.add_argument('--seed_monotonic', default = 1 ,type= int ,  help='gpu device')
parser.add_argument('--off_set', default = .1 ,type= float ,  help='gpu device')
parser.add_argument('--J', default = 10 ,type= int ,  help='gpu device')
parser.add_argument('--L', default = .5 ,type= float ,  help='gpu device')
parser.add_argument('--transform',  default = 'monotonic_fourier' ,type= str ,  help='gpu device')
parser.add_argument('--iteration_min',  default = 500 ,type= int,  help='gpu device')



args = parser.parse_args()
args = make_flags(args,args.config)
exp = Trainer(args)


exp.params_loop()
#test_acc = exp.test()
print('Training completed!')







