from cycler import cycler
import numpy as np

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import json
#plt.xkcd()


plt.rc('axes', prop_cycle=(cycler(color=['r', 'g', 'purple', 'orchid', 'lightblue'])))

on_sshfs = True

model_map = {
    'dcgan': 'DCGAN',
    'dcgan-sn_gp-10_lr-2e5': 'DCGAN-SN-GP',
    'dcgan-sn': 'DCGAN-SN',
    'resnet_gp-10': 'ResNet-GP',
    'resnet-sn_gp-10': 'ResNet-SN-GP'
}


def sshfs_fp(path):
    if on_sshfs:
        return os.path.join('swc_sshfs', path)
    return path

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
    print(os.listdir(path))



if __name__ == '__main__':


    d_types = ['dcgan',
                'dcgan-sn',
                'dcgan-sn_gp-10_lr-2e5',
                'resnet_gp-10',
                'resnet-sn_gp-10'
                ]
    fid_plot(d_types)

    seed = 0
    Z_path = os.path.join('logs', 'dcgan-sn', 'fids', 'samples')
    single_image(Z_path, seed)






