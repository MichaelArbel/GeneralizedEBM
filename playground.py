
import numpy as np

import os


import matplotlib.pyplot as plt
import json


on_sshfs = True

model_map = {
    'dcgan': 'DCGAN',
    'dcgan-sn_gp2-10_lr-2e5': 'DCGAN-SN-GP',
    'dcgan-sn': 'DCGAN-SN',
    'resnet-sn_gp2-10': 'ResNet-SN-GP'
}



import matplotlib as mpl
mpl.rc('font', family='Helvetica Neue')

def main():

    if on_sshfs:
        fname = os.path.join('logs', 'playground', 'fids', 'lmc_fids_1041546.json')
    else:
        fname = os.path.join('swc_sshfs', 'logs', 'playground', 'fids', 'lmc_fids_1041546.json')
    print(fname)

    with open(fname, 'r') as f:
        fid_data = json.load(f)

    fid_train, fid_test = list(zip(*fid_data))

    x_range = np.arange(0, 300, step=10)

    # plt.plot(x_range, fid_train, '.', linestyle='-', label=model_map[d], lw=2)

    plt.plot(x_range, fid_test, '.', linestyle='-.', label='lol', lw=2)

    plt.plot(x_range, np.ones_like(x_range) * fid_test[0], linestyle='dotted', c='gray', lw=1)

    plt.xlabel('number of LMC steps')
    plt.ylabel('FID score')
    plt.ylim([30,300])

    plt.yticks(np.arange(30, 300, 10))

    plt.grid(which='major', axis='y', linewidth=0.5)
    plt.minorticks_on()

    plt.legend()
    plt.title('FID improvement with Langevin dynamics')

    plt.tight_layout()

    plt.savefig('figures/lmc_v_fid.png')

    plt.close()



if __name__ == '__main__':


    d_types = ['dcgan',
                'dcgan-sn',
                'dcgan-sn_gp2-10_lr-2e5',
                'resnet-sn_gp2-10'
                ]
    main()