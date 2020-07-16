# -*- coding: utf-8 -*-
#
# Developed by Zhouxiang Hu <huzhouxiang@mail.ru>

from mxnet import nd
from mxnet.gluon import nn


class Merger_hybrid(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Merger_hybrid,self).__init__(**kwargs)
        self.layer1 = nn.HybridSequential()
        self.layer1.add(
            nn.Conv3D(16, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=16),
            nn.LeakyReLU(.2)
        )
        self.layer2 = nn.HybridSequential()
        self.layer2.add(
            nn.Conv3D(8, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=8),
            nn.LeakyReLU(.2)
        )
        self.layer3 = nn.HybridSequential()
        self.layer3.add(
            nn.Conv3D(4, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=4),
            nn.LeakyReLU(.2)
        )
        self.layer4 = nn.HybridSequential()
        self.layer4.add(
            nn.Conv3D(2, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=2),
            nn.LeakyReLU(.2)
        )
        self.layer5 = nn.HybridSequential()
        self.layer5.add(
            nn.Conv3D(1, kernel_size=3, padding=1),
            nn.BatchNorm(in_channels=1),
            nn.LeakyReLU(.2)
        )
    def hybrid_forward(self, F, raw_feature):
        # print(raw_feature.size())       # nd.Size([batch_size, 9, 32, 32, 32])
        volume_weight = self.layer1(raw_feature)
        #print(volume_weight.shape)     # nd.Size([batch_size, 16, 32, 32, 32])
        volume_weight = self.layer2(volume_weight)
        # print(volume_weight.size())     # nd.Size([batch_size, 8, 32, 32, 32])
        volume_weight = self.layer3(volume_weight)
        # print(volume_weight.size())     # nd.Size([batch_size, 4, 32, 32, 32])
        volume_weight = self.layer4(volume_weight)
        # print(volume_weight.size())     # nd.Size([batch_size, 2, 32, 32, 32])
        volume_weight = self.layer5(volume_weight)
        # print(volume_weight.size())     # nd.Size([batch_size, 1, 32, 32, 32])

        #volume_weight = nd.squeeze(volume_weight, dim=1)
        # print(volume_weight.size())     # nd.Size([batch_size, 32, 32, 32])

        #volume_weights = nd.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        #volume_weights = nd.softmax(volume_weights, dim=1)
        # print(volume_weights.size())        # nd.Size([batch_size, n_views, 32, 32, 32])
        # print(coarse_volumes.size())        # nd.Size([batch_size, n_views, 32, 32, 32])
        #coarse_volumes = coarse_volumes * volume_weights
        #coarse_volumes = nd.sum(coarse_volumes, dim=1)
        return volume_weight
