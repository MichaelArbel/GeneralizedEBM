import numpy as np

import torch
from torch import nn
import math

import torch.nn.functional as F

# official implementation
from torch.nn.utils import spectral_norm as sn_official
# implementation taken from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
# from spectral_normalization.spectral_normalization import SpectralNorm as sn_online

# use the official one beca
spectral_norm = sn_official



class Discriminator(nn.Module):
    def __init__(self, nn_type='dcgan', **kwargs):
        super().__init__()

        self.nn_type = nn_type

        if nn_type == 'dcgan':
            # adapted from pytorch website
            # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#implementation

            # nc = number of input channels (input image size square)
            # ndf = number of filters, state sizes

            # defaults
            nc = 3
            ndf = 64
            leak = 0.2

            if 'nc' in kwargs:
                nc = kwargs['nc']
            if 'ndf' in kwargs:
                ndf = kwargs['ndf']
            if 'leak' in kwargs:
                leak = kwargs['leak']

            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation...)
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf*8) x 4 x 4
                #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                # change documented at https://github.com/pytorch/examples/issues/486
                nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
                nn.Sigmoid()
            )

        elif nn_type == 'dcgan-ns':
            nc = 3
            ndf = 64
            leak = 0.2

            if 'nc' in kwargs:
                nc = kwargs['nc']
            if 'ndf' in kwargs:
                ndf = kwargs['ndf']
            if 'leak' in kwargs:
                leak = kwargs['leak']

            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation...)
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(leak, inplace=True),
                # state size. (ndf*8) x 4 x 4
                #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                # change documented at https://github.com/pytorch/examples/issues/486
                nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False)
                #nn.Sigmoid()
            )

        elif nn_type == 'dcgan-sn':
            # adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
            # with spectral norm from pytorch

            # defaults
            nc = 3
            ndf = 64
            leak = 0.1
            w_g = 4

            if 'nc' in kwargs:
                nc = kwargs['nc']
            if 'ndf' in kwargs:
                ndf = kwargs['ndf']
            if 'leak' in kwargs:
                leak = kwargs['leak']

            self.main = nn.Sequential(
                # layer 1
                spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=True)),
                nn.LeakyReLU(leak),
                # layer 2
                spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=True)),
                nn.LeakyReLU(leak),
                #layer 3
                spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=True)),
                nn.LeakyReLU(leak),
                # layer 4
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=True)),
                nn.LeakyReLU(leak),
                # layer 5
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True)),
                nn.LeakyReLU(leak),
                # layer 6
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True)),
                nn.LeakyReLU(leak),
                # layer 7
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True)),
                nn.LeakyReLU(leak),
                nn.Flatten(),
                spectral_norm(nn.Linear(w_g * w_g * 512, 1))
            )

        elif nn_type == 'resnet-sn':
            # adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
            # with spectral norm from pytorch

            nc = 3
            self.disc_size = 128

            self.fc = nn.Linear(self.disc_size, 1)
            nn.init.xavier_uniform_(self.fc.weight.data, 1.)

            self.main = nn.Sequential(
                FirstResBlockDiscriminator(nc, self.disc_size, stride=2, sn=1),
                ResBlockDiscriminator(self.disc_size, self.disc_size, stride=2, sn=1),
                ResBlockDiscriminator(self.disc_size, self.disc_size, sn=1),
                ResBlockDiscriminator(self.disc_size, self.disc_size, sn=1),
                nn.ReLU(),
                nn.AvgPool2d(8),
                nn.Flatten(),
                spectral_norm(self.fc)
            )

        elif nn_type == 'resnet':
            nc = 3
            self.disc_size = 128

            self.fc = nn.Linear(self.disc_size, 1)
            nn.init.xavier_uniform_(self.fc.weight.data, 1.)

            self.main = nn.Sequential(
                FirstResBlockDiscriminator(nc, self.disc_size, stride=2, sn=0),
                ResBlockDiscriminator(self.disc_size, self.disc_size, stride=2, sn=0),
                ResBlockDiscriminator(self.disc_size, self.disc_size, sn=0),
                ResBlockDiscriminator(self.disc_size, self.disc_size, sn=0),
                nn.ReLU(),
                nn.AvgPool2d(8),
                nn.Flatten(),
                self.fc
            )

        else:
            raise NotImplementedError()
            


    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)





### helpers


# for the spectral resnet

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, sn=1):
        super(ResBlockDiscriminator, self).__init__()

        if sn == 1:
            spec_norm = spectral_norm
        else:
            def spec_norm(x):
                return x

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spec_norm(self.conv1),
                nn.ReLU(),
                spec_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spec_norm(self.conv1),
                nn.ReLU(),
                spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                spec_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, sn=1):
        super(FirstResBlockDiscriminator, self).__init__()

        if sn == 1:
            spec_norm = spectral_norm
        else:
            def spec_norm(x):
                return x

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spec_norm(self.conv1),
            nn.ReLU(),
            spec_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


# def weight_bar(w, u, pi):
#     # _, sigma, _ = torch.svd(w)
#     # sigma = sigma[0]
#
#     w_mat = w.data.view(w.data.shape[0], -1)
#
#     for _ in range(pi):
#         v = l2normalize(torch.mv(torch.t(w_mat), u))
#
#         u = l2normalize(torch.mv(w_mat, v))
#
#     sigma = torch.dot(torch.mv(torch.t(w_mat), u), v)
#     w_bar = w / sigma
#
#     return w_bar, u, sigma

#
# class SpectralNorm(torch.nn.Module):
#
#     def __init__(self, out_features, power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.power_iterations = power_iterations
#         self.out_features = out_features
#         # self.register_buffer("u", torch.randn(out_features, requires_grad=False))
#
#         self.register_buffer("u", torch.randn((1, out_features), requires_grad=False))
#
#     def forward(self, w):
#         w_mat = w.view(w.data.shape[0], -1)
#
#         # with torch.no_grad():
#         #     _, sigma, _ = torch.svd(w_mat)
#         #     sigma = sigma[0]
#
#         #
#         u = self.u
#         with torch.no_grad():
#             for _ in range(self.power_iterations):
#                 v = l2normalize(torch.mm(u, w_mat.data))
#
#                 u = l2normalize(torch.mm(v, torch.t(w_mat.data)))
#
#                 # v = l2normalize(torch.mv(torch.t(w_mat), self.u))
#
#                 # u = l2normalize(torch.mv(w_mat, v))
#
#         # sigma = u.dot(w_mat.mv(v))
#         sigma = torch.sum(torch.mm(u, w_mat) * v)
#
#         if self.training:
#             self.u = u
#         w_bar = torch.div(w, sigma)
#         # w_bar = w / sigma.expand_as(w.data)
#
#         return w_bar, sigma


def max_singular_value(w_mat, u, power_iterations):

    for _ in range(power_iterations):
        v = l2normalize(torch.mm(u, w_mat.data))

        u = l2normalize(torch.mm(v, torch.t(w_mat.data)))

    sigma = torch.sum(torch.mm(u, w_mat) * v)

    return u, sigma, v



class Linear(torch.nn.Linear):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_features), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)


    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_features, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)

            # w_bar = torch.div(w_mat, sigma)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
            # self.w_bar = w_bar.detach()
            # self.sigma = sigma.detach()
        else:
            w_bar = self.weight
        return F.linear(input, w_bar, self.bias)


class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_channels), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_channels, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.conv2d(input, w_bar, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Embedding(torch.nn.Embedding):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.num_embeddings), requires_grad=False))
        else:
            self.register_buffer("u", None)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.num_embeddings, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.embedding(
            input, w_bar, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class Block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 kernel_size=3, stride=1, padding=1, optimized=False, spectral_norm=1):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.optimized = optimized
        self.hidden_channels = out_channels if not hidden_channels else hidden_channels

        self.conv1 = Conv2d(self.in_channels, self.hidden_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.conv2 = Conv2d(self.hidden_channels, self.out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.s_conv = None
        torch.nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        if self.in_channels != self.out_channels or optimized:
            self.s_conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0,
                                 spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.s_conv.weight.data, 1.)

        self.activate = torch.nn.ReLU()

    def residual(self, input):
        x = self.conv1(input)
        x = self.activate(x)
        x = self.conv2(x)
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        if self.s_conv:
            x = self.s_conv(x)
        return x


    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r


class Dblock(Block):

    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, stride=1, padding=1,
                 downsample=False, spectral_norm=1):
        super(Dblock, self).__init__(in_channels, out_channels, hidden_channels, kernel_size, stride, padding,
                                     downsample, spectral_norm)
        self.downsample = downsample

    def residual(self, input):
        x = self.activate(input)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.s_conv:
            x = self.s_conv(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r

class BaseDiscriminator(torch.nn.Module):

    def __init__(self, in_ch, out_ch=None, n_categories=0, l_bias=True, spectral_norm=1):
        super(BaseDiscriminator, self).__init__()
        self.activate = torch.nn.ReLU()
        self.ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.n_categories = n_categories
        self.blocks = torch.nn.ModuleList([Block(3, self.ch, optimized=True, spectral_norm=spectral_norm)])
        self.l = Linear(self.out_ch, 1, l_bias, spectral_norm_pi=spectral_norm)
        torch.nn.init.xavier_uniform_(self.l.weight.data, 1.)
        if n_categories > 0:
            self.l_y = Embedding(n_categories, self.out_ch, spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.l_y.weight.data, 1.)

    def forward(self, input, y=None):
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        x = torch.sum(x, (2, 3))
        output = self.l(x)
        if y is not None:
            w_y = self.l_y(y)
            output += torch.sum(w_y*x, dim=1, keepdim=True)
        return output


class ResnetDiscriminator64(BaseDiscriminator):

    def __init__(self, ch=64, n_categories=0, spectral_norm=0):
        super(ResnetDiscriminator64, self).__init__(ch, ch*16, n_categories, spectral_norm=spectral_norm)
        self.blocks.append(Dblock(self.ch, self.ch*2, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*2, self.ch*4, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*4, self.ch*8, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*8, self.ch*16, downsample=True, spectral_norm=spectral_norm))

