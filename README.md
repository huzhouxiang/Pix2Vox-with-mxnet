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
|          | images | predicted | GT  |
|----------|:----:|:---:|:---:|
aeroplane | ![aeroplane:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/aeroplane/00.png)| ![aeroplane (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/aeroplane/voxels-000011.png)| ![aeroplane (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/aeroplane/voxels-000022.png)|
bench | ![bench:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/bench/00.png)| ![bench (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/bench/voxels-000011.png)| ![bench (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/bench/voxels-000022.png)|
cabinet | ![cabinet:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/cabinet/00.png)| ![cabinet (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/cabinet/voxels-000011.png)| ![cabinet (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/cabinet/voxels-000022.png)|
car | ![car:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/car/00.png)| ![car (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/car/voxels-000011.png)| ![car (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/car/voxels-000022.png)|
chair | ![chair:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/chair/00.png)| ![chair (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/chair/voxels-000011.png)| ![chair (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/chair/voxels-000022.png)|
display | ![display:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/display/00.png)| ![display (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/display/voxels-000011.png)| ![display (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/display/voxels-000022.png)|
lamp | ![lamp:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/lamp/00.png)| ![lamp (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/lamp/voxels-000011.png)| ![lamp (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/lamp/voxels-000022.png)|
speaker | ![speaker:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/speaker/00.png)| ![speaker (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/speaker/voxels-000011.png)| ![speaker (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/speaker/voxels-000022.png)|
rifle | ![rifle:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/rifle/00.png)| ![rifle (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/rifle/voxels-000011.png)| ![rifle (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/rifle/voxels-000022.png)|
sofa | ![sofa:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/sofa/00.png)| ![sofa (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/sofa/voxels-000011.png)| ![sofa (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/sofa/voxels-000022.png)|
table | ![table:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/table/00.png)| ![table (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/table/voxels-000011.png)| ![table (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/table/voxels-000022.png)|
telephone | ![telephone:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/telephone/00.png)| ![telephone (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/telephone/voxels-000011.png)| ![telephone (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/telephone/voxels-000022.png)|
watercraft | ![watercraft:](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/watercraft/00.png)| ![watercraft (generated):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/watercraft/voxels-000011.png)| ![watercraft (GT):](https://github.com/huzhouxiang/Pix2Vox-with-mxnet/blob/master/visualized%20results%20of%20prediction/watercraft/voxels-000022.png)|




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
