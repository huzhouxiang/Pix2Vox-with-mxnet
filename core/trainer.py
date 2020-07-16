# -*- coding: utf-8 -*-
#
# Developed by Hu Zhouxiang <huzhouxiang@mail.ru>

import os
import random
import mxnet as mx
from mxnet import nd,autograd
from mxnet.gluon import nn,data as gdata,loss as gloss,utils as gutils,Trainer,model_zoo,Parameter
import utils.own_dataloader
import utils.binvox_visualization
import utils.data_transforms
import utils.network_utils
import d2lzh as d2l
from config import cfg

from datetime import datetime as dt
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.test import test_net,forward
from models.encoder import Encoder_hybrid
from models.decoder import Decoder_hybrid
from models.refiner import Refiner_hybrid
from models.merger import Merger_hybrid

# set train_transforms	
IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
train_transforms = utils.data_transforms.Compose([
    utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
    utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
    utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
    utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
    utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    utils.data_transforms.RandomFlip(),
    utils.data_transforms.RandomPermuteRGB(),
    utils.data_transforms.ToTensor(),
])

val_transforms = utils.data_transforms.Compose([
    utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
    utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    utils.data_transforms.ToTensor(),
])


# Set up data loader
train_dataset = utils.own_dataloader.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET]('Train', train_transforms,3,None)


train_data_loader = gdata.DataLoader(train_dataset,batch_size=cfg.CONST.BATCH_SIZE,
                                                    shuffle=1,
                                                    last_batch="discard",
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True)
val_dataset = utils.own_dataloader.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET]('Val', val_transforms,3,None)
val_data_loader = gdata.DataLoader(val_dataset,batch_size=1,
                                                    shuffle=1,
                                                    last_batch="discard",
                                                    num_workers=1,
                                                    pin_memory=True)

# Set up networks
encoder = Encoder_hybrid()
decoder = Decoder_hybrid()
refiner = Refiner_hybrid()
merger = Merger_hybrid()
print('[DEBUG] %s number of parameters:%d.' % (dt.now(),utils.network_utils.count_parameters(encoder)))
print('[DEBUG] %s number of parameters:%d.' % (dt.now(),utils.network_utils.count_parameters(decoder)))
print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))
print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))

# Initialize weights of networks
ctx = d2l.try_gpu()
encoder.apply(utils.network_utils.init_weights)
decoder.apply(utils.network_utils.init_weights)
refiner.apply(utils.network_utils.init_weights)
merger.apply(utils.network_utils.init_weights)
encoder.collect_params().reset_ctx(ctx)
decoder.collect_params().reset_ctx(ctx)
merger.collect_params().reset_ctx(ctx)
refiner.collect_params().reset_ctx(ctx)

#set up trianers
if cfg.TRAIN.POLICY == 'adam':
    encoder_trainer = Trainer(encoder.collect_params('.*^(?!vgg0)'),'adam',# not all layers require upgrade parameters
                                   	{"learning_rate":cfg.TRAIN.ENCODER_LEARNING_RATE})
    decoder_trainer = Trainer(decoder.collect_params(),'adam',
                                 	{"learning_rate":cfg.TRAIN.DECODER_LEARNING_RATE})
    refiner_trainer = Trainer(refiner.collect_params(),'adam',{"learning_rate":cfg.TRAIN.ENCODER_LEARNING_RATE})
    merger_trainer = Trainer(merger.collect_params(), 'adam',{"learning_rate":cfg.TRAIN.ENCODER_LEARNING_RATE})
elif cfg.TRAIN.POLICY == 'sgd':
    encoder_trainer = Trainer(encoder.collect_params('.*^(?!vgg0)'),'sgd',# not all layers require upgrade parameters
                                        {"learning_rate":cfg.TRAIN.ENCODER_LEARNING_RATE})
    decoder_trainer = Trainer(decoder.collect_params(),'sgd',
                                 	{"learning_rate":cfg.TRAIN.ENCODER_LEARNING_RATE,"momentum":cfg.TRAIN.MOMENTUM})
    refiner_trainer = Trainer(refiner.collect_params(),'sgd',
                                 	{"learning_rate":cfg.TRAIN.ENCODER_LEARNING_RATE,"momentum":cfg.TRAIN.MOMENTUM})
    merger_trainer = Trainer(merger.collect_params(), 'sgd',
                                	{"learning_rate":cfg.TRAIN.ENCODER_LEARNING_RATE,"momentum":cfg.TRAIN.MOMENTUM})

# Set up loss functions
bce_loss = gloss.SigmoidBinaryCrossEntropyLoss()
# initialize parameters of training
init_epoch = 0
best_iou = -1
best_epoch = -1
batch_size = cfg.CONST.BATCH_SIZE
# Load pretrained model if exists
if cfg.TRAIN.RESUME_TRAIN:
    print('[INFO] %s Recovering weights' % dt.now())
    checkpoint = nd.load("check_point")
    init_epoch = checkpoint[0][0].asscalar()
    best_iou = checkpoint[0][1].asscalar()
    best_epoch = checkpoint[0][2].asscalar()
    encoder.load_parameters('encoder_params')
    decoder.load_parameters('decoder_params')
    encoder_trainer.load_states("encoder_trainer")
    decoder_trainer.load_states("decoder_trainer")
    if os.path.exists('/home/hzx/my pix2vox model  with refiner/merger_params'):
        merger.load_parameters("merger_params")
        refiner.load_parameters("refiner_params")
        merger_trainer.load_states("merger_trainer")
        refiner_trainer.load_states("refiner_trainer")
    encoder.hybridize()
    decoder.hybridize()
    merger.hybridize()
    print("net has been hybridized")
    print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
          (dt.now(), init_epoch, best_iou, best_epoch))


# Training loop
for epoch_idx in range(int(init_epoch), cfg.TRAIN.NUM_EPOCHES):
    epoch_start_time = time.time()
    #losses  
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()
    if epoch_idx % cfg.TRAIN.ENCODER_LR_MILESTONES[0] == 0:
        encoder_trainer.set_learning_rate(cfg.TRAIN.ENCODER_LEARNING_RATE*cfg.TRAIN.GAMMA)
        decoder_trainer.set_learning_rate(cfg.TRAIN.DECODER_LEARNING_RATE*cfg.TRAIN.GAMMA)
        merger_trainer.set_learning_rate(cfg.TRAIN.ENCODER_LEARNING_RATE*cfg.TRAIN.GAMMA)
        refiner_trainer.set_learning_rate(cfg.TRAIN.DECODER_LEARNING_RATE*cfg.TRAIN.GAMMA)
    n_batches = len(train_data_loader)

    for batch_idx,(idx,rendering_images,ground_truth_volumes) in enumerate(train_data_loader):
        # Measure data time
        # Get data from data loader
        rendering_images = rendering_images.as_in_context(ctx)
        ground_truth_volumes = ground_truth_volumes.as_in_context(ctx)
        # Train the encoder, decoder, refiner, and merger
        if batch_idx ==1 and epoch_idx==0:
                encoder.hybridize()
                decoder.hybridize()
                merger.hybridize()
                print("net was hybridized") 
		# hybridization makes training faster
        with autograd.record():
            gen_volumes2,gen_volumes = forward(encoder,decoder,merger,refiner,rendering_images)
            gen_volumes = nd.mean(gen_volumes,axis=1) 
            encoder_loss = bce_loss(gen_volumes,ground_truth_volumes)*10   
            refiner_loss = bce_loss(gen_volumes2, ground_truth_volumes)*10
        autograd.backward([encoder_loss,refiner_loss])
        encoder_trainer.step(batch_size)
        decoder_trainer.step(batch_size)
        refiner_trainer.step(batch_size)
        merger_trainer.step(batch_size)
        
        # Append loss to average metrics
        encoder_losses.update(encoder_loss.mean().asscalar())
        refiner_losses.update(refiner_loss.mean().asscalar())
        #end of one batch
    print("[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f" %
          (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, time.time()-epoch_start_time, encoder_losses.avg,
           refiner_losses.avg))

    # Update Rendering Views
    if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
 	#using random number of views to train the net
        n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
        train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
        print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
              (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

    # Validate the training models
    iou = test_net(cfg, epoch_idx + 1,None,val_data_loader, None, encoder, decoder,refiner,merger)
    if epoch_idx == 0:
        best_iou = 0.0
        best_epoch = 1
    if iou > best_iou:
        best_iou = iou
        best_epoch = epoch_idx + 1
    if iou > best_iou*0.85:
        volume,_ = forward(encoder,decoder,merger,refiner,val_dataset[0][1].expand_dims(axis=0).as_in_context(ctx))
	#if current iou is bigger than 85% of best iou, generate the 3D model and save it in fold generated_models_with_refiner.
        utils.binvox_visualization.get_volume_views(volume,"/home/hzx/my pix2vox model3/generated_models_with_refiner",epoch_idx)
    # Save weights to file
    if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
        utils.network_utils.save_checkpoints(cfg, epoch_idx, encoder, encoder_trainer, decoder, decoder_trainer,best_iou, best_epoch,refiner=refiner,
                     refiner_solver=refiner_trainer, merger=merger, merger_solver=merger_trainer)

