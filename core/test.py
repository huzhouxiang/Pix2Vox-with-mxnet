# -*- coding: utf-8 -*-
#
# Developed by Hu Zhouxiang <huzhouxiang@mail.ru>
import os
import random
import json
import mxnet as mx
from mxnet import nd,autograd
from mxnet.gluon import nn,data as gdata,loss as gloss,utils as gutils,Trainer,model_zoo,Parameter
import utils.data_loaders
import utils.binvox_visualization
import utils.data_transforms
import utils.network_utils
import d2lzh as d2l
from config import cfg
from datetime import datetime as dt
from time import time


from models.encoder import Encoder_hybrid
from models.decoder import Decoder_hybrid
from models.refiner import Refiner_hybrid
from models.merger import Merger_hybrid

def forward(encoder,decoder,merger,refiner,data):
    #start_time = time.time()
    data = data.transpose((1,0,2,3,4))
    features = []
    for i in range(data.shape[0]):
        feature = encoder(data[i])
        features.append(feature)
    #print("shape after encoder:",len(features),features[0])
    raw_features=[]
    gen_volumes= None
    volume_weights = None
    for feature in features:
        feature = feature.reshape(-1,2048,2,2,2)
        raw_feature,gen_volume = decoder(feature)
        raw_feature = nd.concat(raw_feature, gen_volume, dim=1)
        gen_volume = gen_volume.transpose((1,0,2,3,4))
        raw_features.append(raw_feature)
        if gen_volumes is None:
            gen_volumes = gen_volume
        else:
            gen_volumes = nd.concat(gen_volumes,gen_volume,dim=0)
        #print("raw_features.shape",raw_feature.shape)
        volume_weight = merger(raw_feature)#volume_weught.shape = (batch_size,1,32,32,32)
        #print("volume_weight.shape:",volume_weight.shape)
        if volume_weights is None:
            volume_weights = volume_weight
        else:
            volume_weights = nd.concat(volume_weights,volume_weight,dim=1)
    #print("volumes_weight.shape:",volume_weights.shape,gen_volumes.shape)
    volume_weights = nd.softmax(volume_weights,axis=1).transpose((1,0,2,3,4)) 
    #print("volumes_weight.shape:",volume_weights.shape) #shape = (3,16,32,32,32)
    gen_volumes = volume_weights*gen_volumes
    #print(len(gen_volumes),gen_volumes[0].shape)
    gen_volumes2 = gen_volumes[0].expand_dims(axis=0)
    #print(gen_volumes2.shape)
    for i in range(1,len(gen_volumes)):
        gen_volumes2 = nd.concat(gen_volumes2,gen_volumes[i].expand_dims(axis=0),dim=0)
    #print(gen_volumes2.shape)
    gen_volumes2 = nd.sum(gen_volumes2,axis=0)
    gen_volumes2 = nd.clip(gen_volumes2,0,1)
    #print("shape after decoder:",len(raw_features),raw_features[0].shape,gen_volumes.shape)
    #print("shape after merger:",gen_volumes2)
    gen_volumes2 = refiner(gen_volumes2)
    #print("shape after refiner",gen_volumes2)
    #print("max:",nd.max(gen_volumes2),"min:",nd.min(gen_volumes))
    #gen_volumes = nd.array(np.asarray(features)).reshape(-1,2048,2,2,2)
    #print("--"*20)
    #print("calculating costs time:",time.time()-start_time)
    return gen_volumes2,gen_volumes.transpose((1,0,2,3,4))


def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):
   
    bce_loss = gloss.SigmoidBinaryCrossEntropyLoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()
    ctx = d2l.try_gpu()
    for sample_idx, (ids,rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        #id_to_name= {'02691156': 'aeroplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'display', '03636649': 'lamp', '03691459': 'speaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone', '04530566': 'watercraft'}
        
        rendering_images = rendering_images.as_in_context(ctx)
        ground_truth_volume = ground_truth_volume.as_in_context(ctx)
        # Test the encoder, decoder, refiner and merger
        gen_volumes2,gen_volumes = forward(encoder,decoder,merger,refiner,rendering_images)
        
        gen_volumes = nd.mean(gen_volumes,axis=1)
        encoder_loss = bce_loss(gen_volumes,ground_truth_volume)*10
        refiner_loss = bce_loss(gen_volumes2, ground_truth_volume)*10

        # Append loss and accuracy to average metrics
        encoder_losses.update(encoder_loss.mean().asscalar())
        refiner_losses.update(refiner_loss.mean().asscalar())

        # IoU per sample
        sample_iou = []
        for th in cfg.TEST.VOXEL_THRESH:
            _volume = (gen_volumes2>th)
            intersection = nd.sum((_volume*ground_truth_volume))
            union = nd.sum(((_volume+ground_truth_volume)>=1))
            sample_iou.append((intersection / union).asscalar())

        
    print('[INFO] %s Test EDLoss = %.4f RLoss = %.4f IoU = %s' %
             (dt.now(), encoder_loss.mean().asscalar(),
               refiner_loss.mean().asscalar(), ['%.4f' % si for si in sample_iou]))
    return nd.array(sample_iou).mean().asscalar()
