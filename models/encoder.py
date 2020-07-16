# -*- coding: utf-8 -*-
#
# Developed by Zhouxiang Hu <huzhouxiang@mail.ru>



from mxnet import nd
import d2lzh as d2l
from mxnet.gluon import nn,model_zoo,Parameter


class Encoder_hybrid(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Encoder_hybrid,self).__init__(**kwargs)
        vgg19 = model_zoo.vision.vgg19(pretrained=True)
        self.vgg = nn.HybridSequential()
        for i in vgg19.features[:27]:
            self.vgg.add(i) 
        self.layer1 = nn.HybridSequential()
        self.layer1.add(
            nn.Conv2D(512,kernel_size =3),
            nn.BatchNorm(),
            nn.ELU())
        self.layer2 = nn.HybridSequential()
        self.layer2.add(
        nn.Conv2D(512,kernel_size =3),
        nn.BatchNorm(),
        nn.ELU(),
        nn.MaxPool2D(pool_size = 3,strides=3))
        self.layer3 = nn.HybridSequential()
        self.layer3.add(
            nn.Conv2D(256,kernel_size =1),
            nn.BatchNorm(),
            nn.ELU())
        for i in self.vgg:
            i.grad_req = 'null'
    def hybrid_forward(self,F,rendering_image):#shape([batch_size,num_views,3,224,224])
        #rendering_images = rendering_images.transpose((1,0,2,3,4))
        #features = []
        #for rendering_image in rendering_images:
            #if rendering_images.shape == (3,224,224):
                #rendering_images=nd.expand_dims(rendering_images,axis=0)
            #print(rendering_images.shape)
        feature = self.vgg(rendering_image)
        #print(features.shape)    # shape([batch_size, 512, 28, 28])
        feature = self.layer1(feature)
        #print(features.shape)    # nd.Size([batch_size, 512, 28, 28])
        feature = self.layer2(feature)
        #print(features.shape)    # nd.Size([batch_size, 512, 26, 26])
        feature = self.layer3(feature)
        #print(features.shape)    # nd.Size([batch_size, 256, 8, 8])
        #features.append(feature)
        return feature
