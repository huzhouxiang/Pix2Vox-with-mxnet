# -*- coding: utf-8 -*-
#
# Developed by Zhouxiang Hu <huzhouxiang@mail.ru>

from mxnet import nd
from mxnet.gluon import nn

class Refiner_hybrid(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Refiner_hybrid,self).__init__(**kwargs)
        self.layer1 = nn.HybridSequential()
        self.layer1.add(
            nn.Conv3D(32, kernel_size=4, padding=2),
            nn.BatchNorm(in_channels=32),
            nn.LeakyReLU(.2),
            nn.MaxPool3D(pool_size=2)
        )
        self.layer2 = nn.HybridSequential()
        self.layer2.add(
            nn.Conv3D(64, kernel_size=4, padding=2),
            nn.BatchNorm(in_channels=64),
            nn.LeakyReLU(.2),
            nn.MaxPool3D(pool_size=2)
        )
        self.layer3 = nn.HybridSequential()
        self.layer3.add(
            nn.Conv3D(128, kernel_size=4, padding=2),
            nn.BatchNorm(in_channels=128),
            nn.LeakyReLU(.2),
            nn.MaxPool3D(pool_size=2)
        )
        self.layer4 = nn.HybridSequential()
        self.layer4.add(
            nn.Dense(2048,activation = 'relu')
        )
        self.layer5 = nn.HybridSequential()
        self.layer5.add(
            nn.Dense(8192,activation='relu')
        )
        self.layer6 = nn.HybridSequential()
        self.layer6.add(
            nn.Conv3DTranspose(64, kernel_size=4, strides=2, padding=1, use_bias=False ),
            nn.BatchNorm(in_channels = 64),
            nn.Activation('relu')
        )
        self.layer7 = nn.HybridSequential()
        self.layer7.add(
            nn.Conv3DTranspose(32, kernel_size=4, strides=2, padding=1, use_bias=False),
            nn.BatchNorm(in_channels =32),
            nn.Activation('relu')
        )
        self.layer8 = nn.HybridSequential()
        self.layer8.add(
            nn.Conv3DTranspose(1, kernel_size=4, strides=2, padding=1, use_bias=False),
            nn.Activation('sigmoid')
        )

    def hybrid_forward(self, F, coarse_volumes):
        volumes_32_l = coarse_volumes.reshape((-1, 1, 32, 32, 32))
        # print(volumes_32_l.size())       #  Size([batch_size, 1, 32, 32, 32])
        volumes_16_l = self.layer1(volumes_32_l)
        # print(volumes_16_l.size())       #  Size([batch_size, 32, 16, 16, 16])
        volumes_8_l = self.layer2(volumes_16_l)
        # print(volumes_8_l.size())        #  Size([batch_size, 64, 8, 8, 8])
        volumes_4_l = self.layer3(volumes_8_l)
        # print(volumes_4_l.size())        #  Size([batch_size, 128, 4, 4, 4])
        flatten_features = self.layer4(volumes_4_l.reshape(-1, 8192))
        # print(flatten_features.size())   #  Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())   #  Size([batch_size, 8192])
        volumes_4_r = volumes_4_l + flatten_features.reshape(-1, 128, 4, 4, 4)
        # print(volumes_4_r.size())        #  Size([batch_size, 128, 4, 4, 4])
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print(volumes_8_r.size())        #  Size([batch_size, 64, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print(volumes_16_r.size())       #  Size([batch_size, 32, 16, 16, 16])
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # Size([batch_size, 1, 32, 32, 32])

        return volumes_32_r.reshape((-1, 32, 32, 32))
