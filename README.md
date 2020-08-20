# Fater-RCNN-in-pytorch-with-BDD100k
Train faster rcnn and evaluate in BDD100k dataset with pytorch.

Dataset：BDD100K

BDD100K(https://bair.berkeley.edu/blog/2018/05/30/bdd/)

Train: 70000，Valid: 10000，Test: 20000（The tag for the test image is not provided）.

Categories include：['bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider']

## Dataset:
   BDD100K数据集是伯克利大学AI实验室（BAIR）发布的大型驾驶视频数据集。BDD100K 数据集包含10万段高清视频，每个视频约40秒。通过每个视频的第10秒对关键帧进行采样，最终得到10万张图片（图片尺寸：1280 * 720 ），其中7万张训练图片，2万张测试图片和1万张验证图片，测试集图片没有对应的图片标签文件。
   
   因拍摄的时间、环境的不同，会呈现出较大的不同，第一张图片灰暗、模糊，而第二张图片强光。

<div align=center>
<img width="40%" src="https://github.com/Cathy-t/Fater-RCNN-in-pytorch-with-BDD100k/blob/master/data/original1.png"/>
<img width="40%" src="https://github.com/Cathy-t/Fater-RCNN-in-pytorch-with-BDD100k/blob/master/data/original2.png"/>
</div>

* 数据集类别信息
>根据给出的数据集标签，类别一共有十类，分别为'bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'. 各类对象数目统计分布如图3所示，图片中car这一类别占的比例最大，train这一类别占的比例最小。
<div align=center><img src="https://github.com/Cathy-t/Fater-RCNN-in-pytorch-with-BDD100k/blob/master/data/distribution.png"/></div>

## 训练后的可视化
<div align=center>
<img width="40%" src="https://github.com/Cathy-t/Fater-RCNN-in-pytorch-with-BDD100k/blob/master/det_images/epoch2/cabc30fc-e7726578.jpg"/>
<img width="40%" src="https://github.com/Cathy-t/Fater-RCNN-in-pytorch-with-BDD100k/blob/master/det_images/epoch2/cabc30fc-eb673c5a.jpg"/>
</div>
