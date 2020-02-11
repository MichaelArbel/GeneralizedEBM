from torch import nn

import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nz=100, nn_type='dcgan', **kwargs):
        super().__init__()

        self.nn_type = nn_type

        nc = 3
        ngf = 64

        # z_dim is latent variable dimension for generator
        self.z_dim = nz

        if nn_type == 'dcgan':
            # adapted from pytorch website
            # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#implementation

            # defaults
            nc = 3
            ngf = 64

            # nc = number of channels
            # ngf = number of generator filters

            if 'nc' in kwargs:
                nc = kwargs['nc']
            if 'ndf' in kwargs:
                ndf = kwargs['ndf']

            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(inplace=False),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(inplace=False),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(inplace=False),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Tanh()
            )

        elif nn_type == 'dcgan-sn':
            # adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
            # with spectral norm from pytorch

            self.main = nn.Sequential(
                #nn.Linear(z_dim,)
                nn.ConvTranspose2d(nz, 512, 4, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, nc, 3, stride=1, padding=(1,1)),
                nn.Tanh() # range between -1 and 1
            )


        elif nn_type == 'resnet-sn':
            # adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
            # with spectral norm from pytorch

            self.gen_size = 128

            nc = 3

            self.dense = nn.Linear(self.z_dim, 4 * 4 * self.gen_size)
            self.final = nn.Conv2d(self.gen_size, nc, 3, stride=1, padding=1)
            nn.init.xavier_uniform_(self.dense.weight.data, 1.)
            nn.init.xavier_uniform_(self.final.weight.data, 1.)

            self.main = nn.Sequential(
                ResBlockGenerator(self.gen_size, self.gen_size, stride=2),
                ResBlockGenerator(self.gen_size, self.gen_size, stride=2),
                ResBlockGenerator(self.gen_size, self.gen_size, stride=2),
                nn.BatchNorm2d(self.gen_size),
                nn.ReLU(),
                self.final,
                nn.Tanh()
            )

        else:
            raise NotImplementedError()


    def forward(self, input):
        if self.nn_type in ['dcgan', 'dcgan-sn']:
            output = self.main(input.view(-1, self.z_dim, 1, 1))
        elif self.nn_type in ['spectral-resnet']:
            output = self.main(self.dense(input).view(-1, self.gen_size, 4, 4))
        
        return output



### helpers

# for spectral_resnet

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)
