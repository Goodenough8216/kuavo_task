# kuavo_task 视觉感知部分    *by Wang Jiancheng*
## 1.项目介绍
`yolo train model=yolo11x.pt epochs=100 batch=8 source=inputvideo.mp4`  
数据集inputvideo  
输出outputvideo.mp4,黄色框为yolo检测结果，红点为结合kalman滤波估计结果  
kuavo_task.py是卡尔曼滤波与YOLO结合目标检测的文件，输入inputvideo,输出outputvideo  
src是ros项目文件，可以发送kalman+YOLO检测结果到/detected_centers话题  
环境配置在requirement.txt  
## 2.如何运行
新建终端于{work_space}/src

`catkin_create_pkg detection rospy std_msgs sensor_msgs cv_bridge geometry_msgs`
(`ultralytics`这个包需要手动在`package.xml`中加入，已经替你添加了)

然后在IDE中手动加入你的pkg下面的所有文件，或者用CLI

接着，新建终端
`roscore`启动ros核

再新建终端
`rosrun detection yolo_node.py`运行节点
或者
`roslaunch detection detection.launch`运行launch

再新建终端
`rostopic echo /detected_centers`查看话题内容

如果幸运的话，你将看到rosrun终端中的运行INFO和rostopic终端中的坐标信息，还有opencv的图像。  
不过我的项目是cpu的torch，运行的可能会很慢，在我的虚拟机中（16个cpu核心），hz大概是0.2:(
## 3.完成的任务——视觉感知
### 3.1跑通yolo模型，完成基础的图像识别功能，将结果发送到话题
### 3.2使用数据集训练，得到.pt文件（演示是在虚拟机中，所以用了cpu版，不过gpu也有）
### 3.3结合卡尔曼滤波，在某帧没有检测到时，或者某帧有噪声时，仍能估计目标位置，并且估计值在合理范围内
