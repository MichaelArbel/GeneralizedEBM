from cycler import cycler
import numpy as np

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import pickle as pkl

import torch
import matplotlib.gridspec as gridspec

#plt.xkcd()


plt.rc('axes', prop_cycle=(cycler(color=['r', 'g', 'purple', 'orchid', 'lightblue'])))

on_sshfs = False

model_map = {
    'dcgan': 'DCGAN',
    'dcgan-sn_gp-10_lr-2e5': 'DCGAN-SN-GP',
    'dcgan-sn': 'DCGAN-SN',
    'resnet_gp-10': 'ResNet-GP',
    'resnet-sn_gp-10': 'ResNet-SN-GP'
}


mpl.rc('font', family='Helvetica Neue')
#mpl.rc('font', family='Humor Sans')
def fid_plot(d_types):
    plt.figure(figsize=(8,7))
    for d in d_types:
        fname = sshfs_fp('logs', d, 'fids', 'posterior_fids.json')
        print(fname)

        with open(fname, 'r') as f:
            fid_data = json.load(f)

        ts, fid_train, fid_test = fid_data
        ts = ts[:len(fid_test)]

        plt.plot(ts, fid_test, '.', linestyle='-', label=model_map[d], lw=2)

    
    plt.axhline(y=fid_test[0], linestyle='dotted', color='gray', linewidth=2, xmin=-100,xmax=100)
    plt.xlabel('number of LMC steps')
    plt.ylabel('FID score')
    pmin = 30
    pmax = 70
    plt.ylim([pmin, pmax])

    plt.yticks(np.arange(pmin, pmax, 1))

    plt.grid(which='major', axis='both', linewidth=0.2)
    plt.minorticks_on()

    plt.legend()
    plt.title('FID changes with Langevin dynamics')

    plt.tight_layout()

    plt.savefig('figures/lmc_v_fid2.png')

    plt.close()


def single_image(path, seed=0):
    files = os.listdir(sshfs_fp(path))
    files.sort()
    samples = []
    fig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(3, 10)
    gs.update(wspace=0.05, hspace=0.05)
    i = 0
    for f in files:
        if f.endswith('.pkl'):
            with open(os.path.join(sshfs_fp(path), f), 'rb') as ff:
                imgs = pkl.load(ff)
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            sample = imgs[seed].detach().numpy()
            sample_t = sample.transpose((1,2,0)) * 0.5 + 0.5
            samples.append(sample_t)
            plt.imshow(sample_t)
            i += 1
            if i >= 30:
                break
    plt.savefig(f'figures/single_image_{seed}.png', bbox_inches='tight')



if __name__ == '__main__':


    d_types = ['dcgan',
                'dcgan-sn',
                'dcgan-sn_gp-10_lr-2e5',
                'resnet_gp-10',
                'resnet-sn_gp-10'
                ]
    #fid_plot(d_types)

    seed = 0
    Z_path = os.path.join('logs', 'dcgan-sn', 'fids', 'samples')
    single_image(Z_path, seed)






