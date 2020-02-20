from cycler import cycler
import numpy as np

import os


import matplotlib.pyplot as plt
import json
plt.xkcd()
plt.rc('axes', prop_cycle=(cycler(color=['r', 'g', 'purple', 'orchid'])))

on_sshfs = True

model_map = {
    'dcgan': 'DCGAN',
    'dcgan-sn_gp-10_lr-2e5': 'DCGAN-SN-GP',
    'dcgan-sn': 'DCGAN-SN',
    'resnet_gp-10': 'ResNet-GP',
    'resnet-sn_gp-10': 'ResNet-SN-GP'
}



import matplotlib as mpl
mpl.rc('font', family='Helvetica Neue')
mpl.rc('font', family='Humor Sans')
def main(d_types):
    plt.figure(figsize=(8,7))
    for d in d_types:
        if on_sshfs:
            fname = os.path.join('logs', d, 'fids', 'posterior_fids.json')
        else:
            fname = os.path.join('swc_sshfs', 'logs', d, 'fids', 'poterior_fids.json')
        print(fname)

        with open(fname, 'r') as f:
            fid_data = json.load(f)

        ts, fid_train, fid_test = fid_data
        ts = ts[:-1]


        # plt.plot(x_range, fid_train, '.', linestyle='-', label=model_map[d], lw=2)

        plt.plot(ts, fid_test, 'o', linestyle='-.', label=model_map[d], lw=3)

    #plt.plot(x_range, np.ones_like(x_range) * fid_test[0], linestyle='dotted', c='gray', lw=1)
    
    plt.axhline(y=fid_test[0], linestyle='dotted', color='gray', linewidth=2, xmin=-100,xmax=100)
    plt.xlabel('number of LMC steps')
    plt.ylabel('FID score')
    plt.ylim([30,37])

    plt.yticks(np.arange(30, 37, 1))

    plt.grid(which='major', axis='y', linewidth=0.2)
    plt.minorticks_on()

    plt.legend()
    plt.title('FID improvement with Langevin dynamics, until it blows up')

    plt.tight_layout()

    plt.savefig('figures/lmc_v_fid2.png')

    plt.close()



if __name__ == '__main__':


    d_types = ['dcgan',
                #'dcgan-sn',
                #'dcgan-sn_gp-10_lr-2e5',
                'resnet_gp-10',
                #'resnet-sn_gp-10'
                ]
    main(d_types)
