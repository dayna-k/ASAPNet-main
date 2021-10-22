"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator', 'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        #print(self.named_children())
        for name, D in self.named_children():
            #print(name)
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        #print(len(result))
        #print(len(result[0]))
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        parser.add_argument('--ndf_max', type=int, default=512, help='maximal number of discriminator filters')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, opt.ndf_max)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleEdgeDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator', 'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)
        for i in range(opt.num_D):
            subnetD_edge = self.create_single_discriminator(opt)
            self.add_module('discriminator_edge_%d' % i, subnetD_edge)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        avgPool = nn.AvgPool2d(4, stride=4)
        small_Gray_input = avgPool(input)
        biup = nn.UpsamplingBilinear2d(scale_factor=4)
        small_Gray_input = biup(small_Gray_input)
        edge = torch.abs(input-small_Gray_input)

        result = []
        result_edge = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        #print(self.named_children())
        for name, D in self.named_children():
            #print(name)
            if name == "discriminator_0" or  name == "discriminator_1":
                #print("discriminator")
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)

            elif name == "discriminator_edge_0" or  name == "discriminator_edge_1":
                #print("discriminator edge")
                out_edge = D(edge)
                if not get_intermediate_features:
                    out_edge = [out_edge]
                result_edge.append(out_edge)
                edge = self.downsample(edge)

        return [result, result_edge]

class Edge_Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""


    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Edge_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # 아래 원래 1채널 아웃풋으로 계산 하지만 나는 진짜로는 segmentation맵을 뽑는게 좋을것같다. Unsupervised로
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        # sequence += [nn.Softmax(1)]
        seg = [nn.Conv2d(3, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seg += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 3, 8)
        seg += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # 아래 원래 1채널 아웃풋으로 계산 하지만 나는 진짜로는 segmentation맵을 뽑는게 좋을것같다. Unsupervised로
        seg += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)
        self.seg = nn.Sequential(*seg)
        self.avgPool = nn.AvgPool2d(4, stride=4)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, input):
        """Standard forward."""
        # input_r = input[:, 0, :, :]
        # input_g = input[:, 1, :, :]
        # input_b = input[:, 2, :, :]
        # Gray_input = (input_r + input_g + input_b) / 3
        # [B,W,H] = Gray_input.shape
        # Gray_input = torch.reshape(Gray_input, [B, 1, W, H])
        
        small_Gray_input = self.avgPool(input)
        small_Gray_input = self.up(small_Gray_input)
        edge = torch.abs(input-small_Gray_input)

        discriminator_output = torch.cat([self.model(input),self.seg(edge)],1)
        # discriminator_output = torch.mean(discriminator_output,1,keepdim=True)
        return discriminator_output



class Seg_Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Seg_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]


        # 아래 원래 1채널 아웃풋으로 계산 하지만 나는 진짜로는 segmentation맵을 뽑는게 좋을것같다. Unsupervised로
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        # sequence += [nn.Softmax(1)]

        seg = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seg += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 3, 8)
        seg += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]


        # 아래 원래 1채널 아웃풋으로 계산 하지만 나는 진짜로는 segmentation맵을 뽑는게 좋을것같다. Unsupervised로
        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        seg += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)
        self.seg = nn.Sequential(*seg)

    def forward(self, input):
        """Standard forward."""

        input_r = input[:, 0, :, :]
        input_g = input[:, 1, :, :]
        input_b = input[:, 2, :, :]
        Gray_input = (input_r + input_g + input_b) / 3
        [B,W,H] = Gray_input.shape
        Gray_input = torch.reshape(Gray_input, [B, 1, W, H])
        # Gray_input = (Gray_input + 1) / 2.0 * 255.0

        # Extract Edge_map
        x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        convy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
        weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

        weights_x = weights_x.cuda()
        weights_y = weights_y.cuda()

        convx.weight = nn.Parameter(weights_x)
        convy.weight = nn.Parameter(weights_y)
        g1_x = convx(Gray_input)
        g1_y = convy(Gray_input)

        g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
        g_1 = torch.reshape(g_1,[B,1,W,H])
        g_1 = torch.cat([g_1,g_1,g_1],1)
        Gradient = self.seg(g_1)
        discriminator_output = torch.cat([self.model(input),Gradient],1)
        discriminator_output = torch.mean(discriminator_output,1,keepdim=True)
        return discriminator_output



class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)