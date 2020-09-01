import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from .GhostNet import ghostnet
import time


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if in_planes>=16:
            self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        else:
            self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# aggregation of the high-level(teacher) features
class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


# aggregation of the low-level(student) features
class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

    def forward(self, x1, x2, x3):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2


# Refinement flow
class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, attention, x1, x2, x3):
        # Note that there is an error in the manuscript. In the paper, the refinement strategy is depicted as ""f'=f*S1"", it should be ""f'=f+f*S1"".
        x1 = x1 + torch.mul(x1, self.upsample4(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        x3 = x3 + torch.mul(x3, attention)

        return x1, x2, x3

# BBSNet
class BBSNet(nn.Module):
    def __init__(self, channel=16):
        super(BBSNet, self).__init__()

        # Backbone model
        self.net = ghostnet()

        # Decoder 1
        self.rfb2_1 = GCM(40, channel)
        self.rfb3_1 = GCM(112, channel)
        self.rfb4_1 = GCM(160, channel)
        self.agg1 = aggregation_init(channel)

        # Decoder 2
        self.rfb0_2 = GCM(16, channel)
        self.rfb1_2 = GCM(24, channel)
        self.rfb5_2 = GCM(40, channel)
        self.agg2 = aggregation_final(channel)

        # upsample function
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Refinement flow
        self.HA = Refine()

        # Components of DEM module
        self.atten_depth_channel_0 = ChannelAttention(3)
        self.atten_depth_channel_1 = ChannelAttention(16)
        self.atten_depth_channel_2 = ChannelAttention(24)
        self.atten_depth_channel_3_1 = ChannelAttention(40)
        self.atten_depth_channel_4_1 = ChannelAttention(112)

        self.atten_depth_spatial_0 = SpatialAttention()
        self.atten_depth_spatial_1 = SpatialAttention()
        self.atten_depth_spatial_2 = SpatialAttention()
        self.atten_depth_spatial_3_1 = SpatialAttention()
        self.atten_depth_spatial_4_1 = SpatialAttention()

        # Components of PTM module
        self.inplanes = 16 * 2
        self.deconv1 = self._make_transpose(TransBasicBlock, 16 * 2, 3, stride=2)
        # self.inplanes = 16
        # self.deconv2 = self._make_transpose(TransBasicBlock, 16, 3, stride=2)
        self.agant1 = self._make_agant_layer(16 * 3, 16 * 2)
        # self.agant2 = self._make_agant_layer(16 * 2, 16)
        self.out0_conv = nn.Conv2d(16 * 3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(16 * 2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(16 * 1, 1, kernel_size=1, stride=1, bias=True)

        cp = []

        cp.append(BasicConv2d(16,3,1))
        cp.append(BasicConv2d(24,16,1))
        cp.append(BasicConv2d(40,24,1))
        cp.append(BasicConv2d(112,40,1))
        cp.append(BasicConv2d(160,112,1))
        self.CP = nn.ModuleList(cp)

        if self.training:
            # self.initialize_weights()
            net = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
            pretrained_dict = net.state_dict()
            self.net.load_state_dict(pretrained_dict,strict=False)


    def forward(self, x, x_depth):
        x_depth = self.net.conv_stem(x_depth)
        x_depth = self.net.bn1(x_depth)
        x_depth = self.net.relu(x_depth)
        x_depth = self.net.layer1(x_depth)
        x1_depth = self.net.layer2(x_depth)
        x2_depth = self.net.layer3(x1_depth)
        x3_1_depth = self.net.layer4(x2_depth)
        x4_1_depth = self.net.layer5(x3_1_depth)


        # layer0 merge
        x_depth = self.CP[0](x_depth)
        temp = x_depth.mul(self.atten_depth_channel_0(x_depth))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        x = x + F.interpolate(temp,scale_factor=2,mode='bilinear',align_corners=True)

        # layer0 merge end

        x = self.net.conv_stem(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.layer1(x)

        # layer1 merge
        x1_depth = self.CP[1](x1_depth)
        temp = x1_depth.mul(self.atten_depth_channel_1(x1_depth))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        x = x + F.interpolate(temp,scale_factor=2,mode='bilinear',align_corners=True)
        # layer1 merge end

        x1 = self.net.layer2(x)  # 24 x 16 x 16

        # layer2 merge
        x2_depth = self.CP[2](x2_depth)
        temp = x2_depth.mul(self.atten_depth_channel_2(x2_depth))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        x1 = x1 + F.interpolate(temp,scale_factor=2,mode='bilinear',align_corners=True)
        # layer2 merge end

        x2 = self.net.layer3(x1)  # 40 x 24 x 24

        # layer3 merge
        x3_1_depth = self.CP[3](x3_1_depth)
        temp = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))
        temp = temp.mul(self.atten_depth_spatial_3_1(temp))
        x2 = x2 + F.interpolate(temp,scale_factor=2,mode='bilinear',align_corners=True)
        # layer3 merge end

        x2_1 = x2
        x3_1 = self.net.layer4(x2_1)  # 112 x 16 x 16

        # layer4 merge
        x4_1_depth = self.CP[4](x4_1_depth)
        temp = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))
        temp = temp.mul(self.atten_depth_spatial_4_1(temp))
        x3_1 = x3_1+ F.interpolate(temp,scale_factor=2,mode='bilinear',align_corners=True)
        # layer4 merge end

        x4_1 = self.net.layer5(x3_1)  # 160 x 8 x 8
        time_2 = time.time()
        # produce initial saliency map by decoder1
        x2_1 = self.rfb2_1(x2_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        attention_map = self.agg1(x4_1, x3_1, x2_1)

        # Refine low-layer features by initial map
        x, x1, x5 = self.HA(attention_map.sigmoid(), x, x1, x2)

        # produce final saliency map by decoder2
        x0_2 = self.rfb0_2(x)
        x1_2 = self.rfb1_2(x1)
        x5_2 = self.rfb5_2(x5)
        y = self.agg2(x5_2, x1_2, x0_2)  # *4

        # PTM module
        y = self.agant1(y)
        y = self.deconv1(y)
        # y = self.agant2(y)
        # y = self.deconv2(y)
        y = self.out1_conv(y)

        #print('Speed: %f %f %fFPS' % ((1 / (time_2 - time_1)), (1 / (time_3 - time_2)), (1 / (time_4 - time_1))))

        return self.upsample(attention_map), y

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    # initialize the weights
    def initialize_weights(self):
        net = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        pretrained_dict = net.state_dict()
        all_params = {}
        for k, v in self.net.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.net.state_dict().keys())
        self.net.load_state_dict(all_params)

        # all_params = {}
        # for k, v in self.net_depth.state_dict().items():
        #     if k == 'conv1.weight':
        #         all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
        #     elif k in pretrained_dict.keys():
        #         v = pretrained_dict[k]
        #         all_params[k] = v
        #     elif '_1' in k:
        #         name = k.split('_1')[0] + k.split('_1')[1]
        #         v = pretrained_dict[name]
        #         all_params[k] = v
        #     elif '_2' in k:
        #         name = k.split('_2')[0] + k.split('_2')[1]
        #         v = pretrained_dict[name]
        #         all_params[k] = v
        # assert len(all_params.keys()) == len(self.net_depth.state_dict().keys())
        # self.net_depth.load_state_dict(all_params)