# -*- coding: utf-8 -*-
#
# Developed by Zhouxiang Hu <huzhouxiang@mail.ru>


from mxnet import nd
from mxnet.gluon import nn



class Decoder_hybrid(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Decoder_hybrid,self).__init__(**kwargs)
        self.layer1 = nn.HybridSequential()
        self.layer1.add(nn.Conv3DTranspose(512,kernel_size=4,strides=2,padding=1,
                        use_bias=False),
                        nn.BatchNorm(in_channels=512),
                        nn.Activation("relu"))
        self.layer2 = nn.HybridSequential()
        self.layer2.add(nn.Conv3DTranspose(128,kernel_size=4,strides=2,padding=1,
                       use_bias=False),
                       nn.BatchNorm(in_channels=128),
                       nn.Activation("relu"))
        self.layer3 = nn.HybridSequential()
        self.layer3.add(nn.Conv3DTranspose(32,kernel_size=4,strides=2,padding=1,
                        use_bias=False),
                        nn.BatchNorm(in_channels=32),
                        nn.Activation("relu"))
        self.layer4 = nn.HybridSequential()
        self.layer4.add(nn.Conv3DTranspose(8,kernel_size=4,strides=2,padding=1,
                        use_bias=False),
                        nn.BatchNorm(in_channels=8),
                        nn.Activation("relu"))
        self.layer5 = nn.HybridSequential()
        self.layer5.add(nn.Conv3DTranspose(1,kernel_size=1,
                        activation="sigmoid",use_bias=False))
    def hybrid_forward(self,F,gen_volumes):
        #print(gen_volumes.shape)   # nd.Size([batch_size, 256, 2, 2, 2])
        gen_volumes = self.layer1(gen_volumes)
        #print(gen_volumes.shape)   # nd.Size([batch_size, 128, 4, 4, 4])
        gen_volumes = self.layer2(gen_volumes)
        #print(gen_volumes.shape)   # nd.Size([batch_size, 64, 8, 8, 8])
        gen_volumes = self.layer3(gen_volumes)
        #print(gen_volumes.shape)   # nd.Size([batch_size, 32, 16, 16, 16])
        gen_volumes = self.layer4(gen_volumes)
        raw_feature = gen_volumes
        #print(gen_volumes.shape)   # nd.Size([batch_size, 8, 32, 32, 32])
        gen_volumes = self.layer5(gen_volumes)
        #print(gen_volumes.shape)   # nd.Size([batch_size, 1, 32, 32, 32])
        #print(raw_feature.shape)  # nd.Size([batch_size, 9, 32, 32, 32])
        #print(gen_volumes.shape)      # nd.Size([batch_size, n_views, 32, 32, 32])
        #print(raw_features.shape)     # nd.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_feature, gen_volumes
