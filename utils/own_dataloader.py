# -*- coding: utf-8 -*-
#
# Developed by Zhouxiang Hu <huzhouxiang@mail.ru>

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
from mxnet import gluon, image,nd
from mxnet.gluon import data as gdata,utils as gutils

from datetime import datetime as dt
from enum import Enum, unique

import utils.binvox_rw


class Shapenet_datasets(gdata.Dataset):
    def __init__(self,datasets_type,transformers,num_views,image_idxes=None):
        self.datasets_type = datasets_type
        self.transformers = transformers
        self.num_views = num_views
        self.image_idxes = image_idxes
        self.file_path = self.get_file_paths()
    def __getitem__(self,idx):
        taxonomy_ids,image_files,voxel_files = self.get_datasets(idx)
        if self.transformers:
            image_files = self.transformers(image_files)
        return taxonomy_ids,image_files,voxel_files;
    def __len__(self):
        return len(self.file_path)    
    def get_file_paths(self):
        with open('/home/hzx/my pix2vox model /datasets/ShapeNet.json') as file:
            rendering_file_names = json.load(file)
        taxonomy_ids = [name["taxonomy_id"] for name in rendering_file_names]
        taxonomy_names = [name["taxonomy_name"] for name in rendering_file_names]
        if self.datasets_type == "Train":
            file_type = "train"
        elif self.datasets_type == "Test":
            file_type = "test"
        elif self.datasets_type == "Val":
            file_type = "val"
        total_dic = []
        for taxonomy_idx,taxonomy_id in enumerate(taxonomy_ids):
            images_file_paths = rendering_file_names[taxonomy_idx][file_type]
            for file_path in images_file_paths:
                if os.path.exists('/home/hzx/Datasets/ShapeNetRendering/%s/%s/rendering'%(taxonomy_id,file_path)):
                    images_path = os.listdir('/home/hzx/Datasets/ShapeNetRendering/%s/%s/rendering'%(taxonomy_id,file_path))
                else:
                    print("path is not exists,ignored")
                    continue
                for num_view in range(self.num_views):
                    image_path = images_path.sort()
                    idx_images_list=[]
                    if self.image_idxes is None:
                        while len(idx_images_list)<self.num_views:
                            idx_images = random.randint(0,len(images_path)-1)
                            if images_path[idx_images] not in idx_images_list and images_path[idx_images][-3:]=="png":
                                idx_images_list.append(images_path[idx_images])
                    else:
                        idx_images_list = [images_path[i] for i in self.image_idxes]
                total_dic.append({
                    "id":taxonomy_id,
                    "image_file_idx":idx_images_list,
                    "file_path":file_path,
                })
            print("completed loading image data and voxel data of (ID = %s,total:%d)"%(taxonomy_id,len(images_file_paths)))
        return total_dic
    def get_datasets(self,idx):
        total_file_path = self.file_path
        taxonomy_id = total_file_path[idx]['id']
        image_file_idx = total_file_path[idx]['image_file_idx']
        file_path = total_file_path[idx]['file_path']
        image_files =[]
        for idex in image_file_idx:
            if os.path.exists('/home/hzx/Datasets/ShapeNetRendering/%s/%s/rendering/%s'
                                   %(taxonomy_id,file_path,idex)):
                image_file = cv2.imread('/home/hzx/Datasets/ShapeNetRendering/%s/%s/rendering/%s'
                                       %(taxonomy_id,file_path,idex),cv2.IMREAD_UNCHANGED).astype("float32")/255
                #if image_files is None:
                    #image_files = image_file.expand_dims(axis=0)
                #else:
                    #image_files = nd.concat(image_files,image_file.expand_dims(axis=0),dim=0)#get data of images
                image_files.append(image_file)
            else:
                print("[WARN] image file not exists")
        volume_path = '/home/hzx/Datasets/ShapeNetVox32/%s/%s/model.binvox'%(taxonomy_id,file_path)
        if os.path.exists(volume_path):
            _,suffix = os.path.splitext(volume_path)
            if suffix==".mat":
                volume = scipy.io.loadmat(volume_path)
                volume = volume['Volume'].astype(np.float32)
            else:
                with open(volume_path, 'rb') as f:
                    volume = utils.binvox_rw.read_as_3d_array(f)
                    volume = volume.data.astype(np.float32)
        else:
            print("[WARN] voxel file not exists") #get data of voxel
        
        return taxonomy_id,image_files,volume




DATASET_LOADER_MAPPING = {
    'ShapeNet': Shapenet_datasets, #here only be used shapenet
    #'Pascal3D': Pascal3dDataLoader,
    #'Pix3D': Pix3dDataLoader
}  # yapf: disable
