from __future__ import print_function

#import torch
import argparse
import yaml

from trainer_gan import Trainer
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
parser.add_argument('--d_path', default = '',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--g_path', default = '',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--load_pre_trained', action = 'store_true', help='log directory for summaries and checkpoints')


parser.add_argument('--dataset', default = 'cifar10',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--data_dir', default = 'data',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_in_file', action = 'store_true' ,  help='gpu device')
parser.add_argument('--no_progress_bar', action = 'store_true' ,  help='gpu device')

parser.add_argument('--num_workers', default = 4 ,type= int,  help='gpu device')


parser.add_argument('--device', default = 0 ,type= int,  help='gpu device')
parser.add_argument('--seed', default = 0 ,type= int ,  help='gpu device')
parser.add_argument('--dtype',  default = '32' ,type= str ,   help='gpu device')

parser.add_argument('--total_epochs', default=100, type=int, help='total number of epochs')

parser.add_argument('--model', default = 'sngan' ,type= str,  help='gpu device')
parser.add_argument('--criterion',  default ='kale' ,type= str ,  help='loss')
parser.add_argument('--latent_noise',  default ='gaussian' ,type= str ,  help='loss')

# Optimizer parameters
parser.add_argument('--optimizer', default = 'adam', type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--b_size', default = 256, type= int,  help='gpu device')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--sgd_momentum', default=0., type=float, help='learning rate')
parser.add_argument('--beta_1', default=0.9, type=float, help='learning rate')
parser.add_argument('--beta_2', default=0.999, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0, type=float, help='learning rate')

parser.add_argument('--lr_decay',  default = 0.9 ,type= float ,  help='gpu device')

parser.add_argument('--n_iter_d_init', default = 100, type= int,  help='gpu device')
parser.add_argument('--n_iter_d', default = 5, type= int,  help='gpu device')
parser.add_argument('--Z_dim', default = 128, type= int,  help='gpu device')



# Scheduler parameters 
parser.add_argument('--use_scheduler', default = 'store_true' ,  help='gpu device')
parser.add_argument('--scheduler',  default ='ExponentialLR' ,type= str ,  help='gpu device')
parser.add_argument('--milestone',  default = '100,200,300' ,type= str ,  help='gpu device')

parser.add_argument('--scheduler_gamma',  default =0.99 ,type= float ,  help='gpu device')

parser.add_argument('--config',  default ='' ,type= str ,  help='gpu device')
parser.add_argument('--with_sacred',  default =False ,type= bool ,  help='gpu device')

parser.add_argument('--fid_samples', default = 1000, type= int,  help='gpu device')



args = parser.parse_args()
args = make_flags(args,args.config)
exp = Trainer(args)

if args.load_pre_trained:
	exp.eval_pre_trained()
else:
	exp.train()
#exp.compute_inception_stats()
#test_acc = exp.test()
print('Training completed!')







