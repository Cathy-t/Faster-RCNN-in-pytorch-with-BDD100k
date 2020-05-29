相关文件及脚本介绍

------------------------------脚本

fasterrcnn_train.py用于模型的训练及保存每一个epoch的模型、
detection_val.py用于加载训练好的模型，并对验证集中的图片生成相应的json文件、
detection_demo.py用于生成可视化目标检测结果、
format_tansfer.py用来对从官网下载的json数据集进行格式的转换，以适应最后的评估函数。

------------------------------文件夹

’data’文件夹下存放经过转换后的用于评估的gt_val.json；
’det_images’文件夹下存放4.1可视化中的结果，即经过detection_demo.py后可视化结果所存放的文件夹；
’models’文件夹中存放3次epoch的model；
’result’文件夹下存放最终经过模型的json生成文件，其实在本次实验中最好的结果为：val_result_3.json；
’tools’文件夹下存放bdd100k数据集的加载以及最终的评估脚本文件evaluate.py，

------------------------------json文件及评估

在上交中的源代码中，由于data数据集过大，这里仅将转换后的用于评估的gt_val.json保留在data文件夹下，

生成的Json格式如下：
[  
   {  
      "name": str,
      "timestamp": 1000,
      "category": str,
      "bbox": [x1, y1, x2, y2],
      "score": float
   }
]
其中name是图片的名字，如c993615f-350c682c，timestamp不用管，固定1000就好，category是检测类别，
[x1,y1]是bounding box左上角的坐标，[x2,y2] 是右下角坐标，score代表置信度。

由于这里的name没有后缀，所以我们在evaluate中进行了读取时的更改，更改行数为：135 136

评估时进入tools文件目录下，使用命令：
python evaluate.py --task det --gt ../data/gt_val.json --result ../result/val_result_xxxx.json