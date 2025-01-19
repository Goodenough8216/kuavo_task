# kuavo_task 视觉感知部分    *by Wang Jiancheng*
## 1.项目介绍
`yolo train model=yolo11x.pt epochs=100 batch=8 source=inputvideo.mp4`  
数据集inputvideo  
输出outputvideo.mp4,黄色框为yolo检测结果，红点为结合kalman滤波估计结果  
kuavo_task.py是卡尔曼滤波与YOLO结合目标检测的文件，输入inputvideo,输出outputvideo  
src是ros项目文件，可以发送kalman+YOLO检测结果到/detected_centers话题  
环境配置在requirement.txt  
## 2.如何运行
