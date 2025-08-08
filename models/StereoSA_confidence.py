from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import cv2
import math
import gc
import time
import timm
from .submodule import *

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class feature_extraction(nn.Module):
    def __init__(self, concat_feature_channel):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(BasicConv(3, 32, bn=True, gelu=True, kernel_size=3, padding=1, stride=2, dilation=1),
                                       BasicConv(32, 32, bn=True, gelu=True, kernel_size=3, padding=1, stride=1, dilation=1),
                                       BasicConv(32, 32, bn=True, gelu=True, kernel_size=3, padding=1, stride=1, dilation=1))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)

        self.lastconv = nn.Sequential(BasicConv(320, 128, bn=True, gelu=True, kernel_size=3, padding=1, stride=1, dilation=1),
                                      nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        concat_feature = self.lastconv(gwc_feature)

        return {"features": [x, l2, l3, l4], "gwc_feature": gwc_feature, "concat_feature": concat_feature}

class upsample(nn.Module):
    def __init__(self, C, cf1, cf2):
        super(upsample, self).__init__()


        self.conv1 = nn.Sequential(BasicConv(1, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(C, C, deconv=True, is_3d=False, bn=True,
                                  gelu=True, kernel_size=4, padding=1, stride=2)

        self.conv2_up = BasicConv(C, C, deconv=True, is_3d=False, bn=True,
                                  gelu=True, kernel_size=4, padding=1, stride=2)

        self.conv1_up = BasicConv(C, 1, deconv=True, is_3d=False, bn=False,
                                  gelu=False, kernel_size=4, padding=1, stride=2)


        self.agg_0 = nn.Sequential(BasicConv(2*C+cf1, C, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(2*C+cf2, C, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1))

        self.spx_4 = nn.Sequential(BasicConv(C+cf2, C, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(C, C, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(C), nn.GELU()
                                   )

        self.dmf = nn.Sequential(BasicConv(1, C, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx = nn.Sequential(nn.ConvTranspose2d(C, 9, kernel_size=4, stride=2, padding=1),)

    def forward(self, left_f1x, left_f2x, init_disp):


        disp_features = self.dmf(init_disp)

        cat_features = self.spx_4(torch.cat((disp_features, left_f2x), dim=1))
        cat_features = self.spx(cat_features)

        sfm = F.softmax(cat_features, 1)

        b, c, h, w = init_disp.shape
        disp_unfold = F.unfold(init_disp, 3, 1, 1).reshape(b, -1, h, w)
        disp_unfold = F.interpolate(disp_unfold, (h * 2, w * 2), mode='nearest').reshape(b, 9, h * 2, w * 2)

        disp = (disp_unfold * sfm).sum(1).unsqueeze(1)
        conv1 = self.conv1(disp)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up[:, 0:conv2.shape[1], 0:conv2.shape[2], 0:conv2.shape[3]], conv2, left_f1x), dim=1)
        conv2 = self.agg_0(conv2)

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1, left_f2x), dim=1)

        conv1 = self.agg_1(conv1)
        conv = self.conv1_up(conv1)

        return conv + disp

class AttFeat(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        group: int = 32,
        ratio: int = 16,) -> None:
        super().__init__()

        self.planes = planes
        self.split_3x3 = BasicConv(inplanes, planes, is_3d=True, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1)
        self.split_5x5 = BasicConv(inplanes, planes, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        d = max(planes // ratio, 32)

        self.fc = nn.Sequential(
            nn.Linear(planes, d),
            nn.BatchNorm1d(d),
            nn.GELU()
        )
        self.fc1 = nn.Linear(d, planes)
        self.fc2 = nn.Linear(d, planes)

        self.u_agg = BasicConv(inplanes, planes, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)

    def forward(self, features_agg: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        batch_size = features_agg.shape[0]
        u1 = self.split_3x3(features_agg)
        u2 = self.split_5x5(features.unsqueeze(2))

        u = u1 + u2

        s = self.avgpool(u).flatten(1)

        z = self.fc(s)
        attn_scores = torch.cat([self.fc1(z), self.fc2(z)], dim=1)
        attn_scores = attn_scores.reshape(batch_size, 2, self.planes)
        attn_scores = attn_scores.softmax(dim=1)

        a = attn_scores[:, 0].reshape(batch_size, self.planes, 1, 1, 1)
        b = attn_scores[:, 1].reshape(batch_size, self.planes, 1, 1, 1)

        u1_n = u1 * a.expand_as(u1)
        u2_n = u2 * b.expand_as(u2)

        x = u1_n + u2_n + u1
        x = self.u_agg(x)

        return x


class aggregation(nn.Module):
    def __init__(self, in_channels):
        super(aggregation, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, gelu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  gelu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  gelu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  gelu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.s8_adjust = BasicConv(64, 48, kernel_size=1, stride=1, padding=0)
        self.attf_8 = AttFeat(48, 48, group=32, ratio=16)

        self.attf_16 = AttFeat(96, 96, group=32, ratio=16)

        self.attf_32 = AttFeat(144, 144, group=32, ratio=16)

        self.attf_16_up = AttFeat(96, 96, group=32, ratio=16)

        self.s8_adjust_up = BasicConv(64, 48, kernel_size=1, stride=1, padding=0)
        self.attf_8_up = AttFeat(48, 48, group=32, ratio=16)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1= self.attf_8(conv1, self.s8_adjust(features[2]))

        conv2 = self.conv2(conv1)
        conv2 = self.attf_16(conv2, features[1])

        conv3 = self.conv3(conv2)
        conv3 = self.attf_32(conv3, features[0])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.attf_16_up(conv2, features[1])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1= self.attf_8_up(conv1, self.s8_adjust_up(features[2]))

        conv = self.conv1_up(conv1)

        return conv

class conf_upsample(nn.Module):
    def __init__(self, C: int, fc: int) -> None:
        super(conf_upsample, self).__init__()


        self.conv1 = BasicConv(1, C, is_3d=False, bn=True, gelu=True, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = BasicConv(C, C, is_3d=False, bn=True, gelu=True, kernel_size=3, padding=1, stride=2, dilation=1)
        self.conv1_up = BasicConv(C, 1, deconv=True, is_3d=False, bn=True, gelu=True, kernel_size=4, padding=1, stride=2)

        self.cm = nn.Sequential(BasicConv(1, C, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.conf_spx_4 = nn.Sequential(BasicConv(C+fc, C, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(C, C, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(C), nn.ReLU()
                                   )

        self.conf_spx = nn.ConvTranspose2d(C, 9, kernel_size=4, stride=2, padding=1)

    def forward(self, left_f1x: torch.Tensor, init_conf: torch.Tensor) -> torch.Tensor:

        conf_features = self.cm(init_conf)
        cat_features = self.conf_spx_4(torch.cat((conf_features, left_f1x), dim=1))
        cat_features = self.conf_spx(cat_features)
        sfm = F.softmax(cat_features, 1)

        b, c, h, w = init_conf.shape
        conf_unfold = F.unfold(init_conf, 3, 1, 1).reshape(b, -1, h, w)
        conf_unfold = F.interpolate(conf_unfold, (h * 2, w * 2), mode='nearest').reshape(b, 9, h * 2, w * 2)

        conf1 = (conf_unfold * sfm).sum(1).unsqueeze(1)
        conf = self.conv1(conf1)
        conf = self.conv2(conf)
        conf = self.conv1_up(conf)

        return conf + conf1


class LAFNet_ESM(nn.Module):
    def __init__(self, C):
        super(LAFNet_ESM, self).__init__()
        self.C = C

        # cost feature extractor
        self.softmax = nn.Softmax(dim=1)
        self.cost_conv1 = nn.Conv2d(7, C, kernel_size=3, padding=1)
        self.cost_bn1 = nn.BatchNorm2d(C)
        self.cost_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.cost_bn2 = nn.BatchNorm2d(C)
        self.cost_conv3 = nn.Conv2d(C, C, kernel_size=1, padding=0)
        self.cost_bn3 = nn.BatchNorm2d(C)

        # disparity feature extractor
        self.disp_conv1 = nn.Conv2d( 1, C, kernel_size=3, padding=1)
        self.disp_bn1 = nn.BatchNorm2d(C)
        self.disp_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.disp_bn2 = nn.BatchNorm2d(C)
        self.disp_conv3 = nn.Conv2d(C, C, kernel_size=1, padding=0)
        self.disp_bn3 = nn.BatchNorm2d(C)

        # image feature extractor
        self.imag_conv1 = nn.Conv2d(48, C, kernel_size=3, padding=1)
        self.imag_bn1 = nn.BatchNorm2d(C)
        self.imag_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.imag_bn2 = nn.BatchNorm2d(C)
        self.imag_conv3 = nn.Conv2d(C, C, kernel_size=1, padding=0)
        self.imag_bn3 = nn.BatchNorm2d(C)

        # cost attention extractor
        self.cost_att_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.cost_att_bn1 = nn.BatchNorm2d(C)
        self.cost_att_conv2 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.cost_att_bn2 = nn.BatchNorm2d(1)

        # disparity attention extractor
        self.disp_att_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.disp_att_bn1 = nn.BatchNorm2d(C)
        self.disp_att_conv2 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.disp_att_bn2 = nn.BatchNorm2d(1)

        # image attention extractor
        self.imag_att_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.imag_att_bn1 = nn.BatchNorm2d(C)
        self.imag_att_conv2 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.imag_att_bn2 = nn.BatchNorm2d(1)

        self.softmax_att = nn.Softmax(dim=1)

        # scale extractor
        self.scale_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.scale_bn1 = nn.BatchNorm2d(C)
        self.scale_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.scale_bn2 = nn.BatchNorm2d(C)
        self.scale_conv3 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.scale_bn3 = nn.BatchNorm2d(1)

        # embedding
        self.embed_conv1 = nn.Conv2d(3 * C, C, kernel_size=3, padding=1)
        self.embed_bn1 = nn.BatchNorm2d(C)
        self.embed_conv2 = nn.Conv2d( C, C, kernel_size=3, padding=0, stride=3)
        self.embed_bn2 = nn.BatchNorm2d(C)

        # predictor
        self.fusion_conv1 = nn.Conv2d(C + 1, C, kernel_size=3, padding=1)
        self.fusion_bn1_iter1 = nn.BatchNorm2d(C)
        self.fusion_bn1_iter2 = nn.BatchNorm2d(C)
        self.fusion_bn1_iter3 = nn.BatchNorm2d(C)
        self.fusion_conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.fusion_bn2_iter1 = nn.BatchNorm2d(C)
        self.fusion_bn2_iter2 = nn.BatchNorm2d(C)
        self.fusion_bn2_iter3 = nn.BatchNorm2d(C)
        self.fusion_conv3 = nn.Conv2d(C,  1, kernel_size=1, padding=0)
        self.fusion_bn3_iter1 = nn.BatchNorm2d(1)
        self.fusion_bn3_iter2 = nn.BatchNorm2d(1)
        self.fusion_bn3_iter3 = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

        self.conf_up2 = conf_upsample(C, 128)
        self.conf_up1 = conf_upsample(C, 32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.scale_bn3.weight, 0)
        nn.init.constant_(self.scale_bn3.bias, 0)


    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, cost, disp, imag, left_f1x, left_f2x, device):
        x = self.softmax(-self.L2normalize(cost) * 100)
        x = torch.topk(x, k=7, dim=1).values

        x = F.relu(self.cost_bn1(self.cost_conv1(x)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        cost_x = F.relu(self.cost_bn3(self.cost_conv3(x)))

        x = F.relu(self.disp_bn1(self.disp_conv1(disp)))
        x = F.relu(self.disp_bn2(self.disp_conv2(x)))
        disp_x = F.relu(self.disp_bn3(self.disp_conv3(x)))

        x = F.relu(self.imag_bn1(self.imag_conv1(imag)))
        x = F.relu(self.imag_bn2(self.imag_conv2(x)))
        imag_x = F.relu(self.imag_bn3(self.imag_conv3(x)))

        x = F.relu(self.cost_att_bn1(self.cost_att_conv1(cost_x)))
        cost_att_x = self.cost_att_bn2(self.cost_att_conv2(x))

        x = F.relu(self.disp_att_bn1(self.disp_att_conv1(disp_x)))
        disp_att_x = self.disp_att_bn2(self.disp_att_conv2(x))

        x = F.relu(self.imag_att_bn1(self.imag_att_conv1(imag_x)))
        imag_att_x = self.imag_att_bn2(self.imag_att_conv2(x))

        att_x = self.softmax_att(torch.cat((cost_att_x, disp_att_x, imag_att_x), 1))

        cost_att_x = att_x[:,0,:,:].unsqueeze(1)
        disp_att_x = att_x[:,1,:,:].unsqueeze(1)
        imag_att_x = att_x[:,2,:,:].unsqueeze(1)

        cost_x = cost_x * (cost_att_x.repeat(1, self.C, 1, 1))
        disp_x = disp_x * (disp_att_x.repeat(1, self.C, 1, 1))
        imag_x = imag_x * (imag_att_x.repeat(1, self.C, 1, 1))

        x = torch.cat((cost_x, disp_x, imag_x), 1)
        feat = F.relu(self.embed_bn1(self.embed_conv1(x)))

        x = F.relu(self.scale_bn1(self.scale_conv1(feat)))
        x = F.relu(self.scale_bn2(self.scale_conv2(x)))
        scale = 2 * self.sigmoid(self.scale_bn3(self.scale_conv3(x)))

        b, c, h, w = disp.size()

        grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        grid_h = torch.tensor(grid_h, dtype=torch.float, requires_grad=False).to(device).repeat(b, 1, 1, 1)
        grid_w = torch.tensor(grid_w, dtype=torch.float, requires_grad=False).to(device).repeat(b, 1, 1, 1)
        grid = torch.cat((grid_w, grid_h), 1).transpose(1, 2).transpose(2, 3)

        scale_t = scale.transpose(1, 2).transpose(2, 3)

        grid_enlarge = torch.zeros([b, 3 * h, 3 * w, 2]).to(device)

        step_x = 2 / (h - 1)
        step_y = 2 / (w - 1)

        grid_enlarge[:,0::3,0::3,:] = grid + torch.cat(((-1) * step_y * scale_t , (-1) * scale_t), 3)
        grid_enlarge[:,0::3,1::3,:] = grid + torch.cat((( 0) * step_y * scale_t , (-1) * scale_t), 3)
        grid_enlarge[:,0::3,2::3,:] = grid + torch.cat(((+1) * step_y * scale_t , (-1) * scale_t), 3)
        grid_enlarge[:,1::3,0::3,:] = grid + torch.cat(((-1) * step_y * scale_t , ( 0) * scale_t), 3)
        grid_enlarge[:,1::3,1::3,:] = grid + torch.cat((( 0) * step_y * scale_t , ( 0) * scale_t), 3)
        grid_enlarge[:,1::3,2::3,:] = grid + torch.cat(((+1) * step_y * scale_t , ( 0) * scale_t), 3)
        grid_enlarge[:,2::3,0::3,:] = grid + torch.cat(((-1) * step_y * scale_t , (+1) * scale_t), 3)
        grid_enlarge[:,2::3,1::3,:] = grid + torch.cat((( 0) * step_y * scale_t , (+1) * scale_t), 3)
        grid_enlarge[:,2::3,2::3,:] = grid + torch.cat(((+1) * step_y * scale_t , (+1) * scale_t), 3)

        feat_enlarge = F.grid_sample(feat, grid_enlarge, align_corners=True)

        feat = F.relu(self.embed_bn2(self.embed_conv2(feat_enlarge)))

        out = torch.zeros([b, c, h, w], dtype=torch.float).to(device) + 0.5

        x = torch.cat((feat, out), 1)

        x = F.relu(self.fusion_bn1_iter1(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter1(self.fusion_conv2(x)))
        out = F.relu(self.fusion_bn3_iter1(self.fusion_conv3(x)))

        x = torch.cat((feat, out), 1)

        x = F.relu(self.fusion_bn1_iter2(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter2(self.fusion_conv2(x)))
        out = F.relu(self.fusion_bn3_iter2(self.fusion_conv3(x)))

        x = torch.cat((feat, out), 1)

        x = F.relu(self.fusion_bn1_iter3(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter3(self.fusion_conv2(x)))
        init_conf = F.relu(self.fusion_bn3_iter3(self.fusion_conv3(x)))

        out2 = self.conf_up2(left_f1x, init_conf)
        out1 = self.conf_up1(left_f2x, out2)

        return self.sigmoid(out1)

class StereoSA_confidence(nn.Module):
    def __init__(self, maxdisp, device):
        super(StereoSA_confidence, self).__init__()
        self.maxdisp = maxdisp
        self.feature = feature_extraction(16)
        self.device = device

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.GELU()
            )

        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.GELU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv(48, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.GELU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv(64, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96), nn.GELU()
            )

        self.stem_32 = nn.Sequential(
            BasicConv(96, 144, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(144, 144, 3, 1, 1, bias=False),
            nn.BatchNorm2d(144), nn.GELU()
            )

        self.conv = BasicConv(320, 160, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1)

        reduction_multiplier = 16

        self.concat_stem = BasicConv(32, reduction_multiplier, deconv=False, is_3d=True, bn=True, gelu=True, kernel_size=3, padding=1, stride=1)

        self.num_groups = 8
        self.group_stem = BasicConv(self.num_groups, self.num_groups, deconv=False, is_3d=True, bn=True, gelu=True, kernel_size=3, padding=1, stride=1)

        self.agg = BasicConv(24, 24, deconv=False, is_3d=True, bn=True, gelu=True, kernel_size=3, padding=1, stride=1)
        self.aggregation_out = aggregation(24)

        self.confidence_net = LAFNet_ESM(32)
        self.upsample_module_4_2 = upsample(32, 64, 64)
        self.upsample_module_2_1 = upsample(32, 64, 64)

    def forward(self, left, right):

        features_left = self.feature(left)
        features_right = self.feature(right)

        match_left = self.desc(self.conv(features_left["gwc_feature"]))
        match_right = self.desc(self.conv(features_right["gwc_feature"]))

        volume_c = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"], self.maxdisp // 4)
        volume_c = self.concat_stem(volume_c)
        volume_g = build_gwc_volume(match_left, match_right, self.maxdisp // 4, self.num_groups)
        volume_g = self.group_stem(volume_g)
        volume = torch.cat((volume_g, volume_c), dim=1)

        volume = self.agg(volume)

        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_32x = self.stem_32(stem_16x)

        cost = self.aggregation_out(volume, features=[stem_32x, stem_16x, stem_8x])

        disp_samples = torch.arange(0, self.maxdisp // 4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp // 4, 1, 1).repeat(cost.shape[0], 1, cost.shape[3], cost.shape[4])
        init_pred = regression_topk(cost.squeeze(1), disp_samples, 2)

        disp_2 = self.upsample_module_4_2(stem_8x, features_left["features"][1], init_pred)
        fea_stem = torch.cat((stem_2x, features_left["features"][0]), 1)
        disp_1 = self.upsample_module_2_1(features_left["features"][1], fea_stem, disp_2)

        conf_out = self.confidence_net(cost.squeeze(1), init_pred, stem_4x, features_left["features"][3], features_left["features"][0],self.device)

        return disp_1.squeeze(1)*4, conf_out.squeeze(1)


