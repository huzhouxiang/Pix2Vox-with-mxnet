# Pix2Vox-with-mxnet
According to the paper: Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images,which was published by Xie, Haozhe and Yao, Hongxun and Sun, Xiaoshuai and Zhou, Shangchen and Zhang, Shengping. https://github.com/hzxie/Pix2Vox#cite-this-work .

With the goal of experiment this work was rewritten and trained on shapenet under framework MXnet.

In this work model is Pix2Vox-A (model with refiner and merger).
When model was training, model was hybridized to speed up training process.

## Averages of IoU:
aeroplane | bench | cabinet | car | chair | display | lamp | speaker | rifle | sofa | table | telephone | watercraft | average
------------ | -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------
0.647 | 0.465 | 0.745 | 0.877 | 0.650 | 0.450 | 0.562 | 0.746 | 0.522 | 0.702 | 0.540 | 0.717 | 0.559 | 0.629

## Visulization:

plan (generated):
![plan (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/generated%20volomes/voxels-000020.png)
plan (GT):
![plan (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/ground%20truth/voxels-000000.png)

bench (generated):
![plan (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/generated%20volomes/voxels-000021.png)
bench (GT):
![plan (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/ground%20truth/voxels-000001.png)

cabinet (generated):
![plan (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/generated%20volomes/voxels-000022.png)
cabinet (GT):
![plan (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/ground%20truth/voxels-000002.png)

car (generated):
![plan (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/generated%20volomes/voxels-000023.png)
car (GT):
![plan (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/ground%20truth/voxels-000003.png)

chair (generated):
![chair (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/generated%20volomes/voxels-000024.png)
chair (GT)
![chair (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/ground%20truth/voxels-000004.png)

lamp (generated):
![chair (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/generated%20volomes/voxels-000026.png)
lamp (GT)
![chair (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/ground%20truth/voxels-000006.png)
