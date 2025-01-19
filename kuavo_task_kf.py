import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

# 模块 1: 卡尔曼滤波器初始化
def create_kalman_filter():
    """
    初始化卡尔曼滤波器
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 状态向量: [x, y, vx, vy], 观测向量: [x, y]
    
    # 状态转移矩阵 (假设匀速运动)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # 观测矩阵
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # 过程噪声协方差矩阵
    kf.Q = np.eye(4) * 0.01
    
    # 观测噪声协方差矩阵
    kf.R = np.eye(2) * 10
    
    # 初始状态协方差矩阵
    kf.P = np.eye(4) * 100
    
    return kf

# 模块 2: 目标检测
def detect_objects(model, frame, conf_threshold, iou_threshold):
    """
    使用YOLO模型进行目标检测
    """
    results = model(frame, conf=conf_threshold, iou=iou_threshold)
    detections = []
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            continue  # No detections in this result
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = box.cls[0].cpu().item()
            detections.append([x1, y1, x2, y2, conf, cls])    
    # if len(detections) == 0:
    #     return None
    return detections

# 模块 3: 卡尔曼滤波更新
def update_kalman_filter(kf, z): # shape(z) = (2,1) 列向量
    """
    更新卡尔曼滤波器
    """
    # 预测
    kf.predict()
    if z is not None: 
        # 如果有观测结果，更新卡尔曼滤波器
        kf.update(z)  # 使用检测到的位置 [x, y]
    
    return kf  

# 模块 4: 绘制检测和跟踪结果
def draw_results(frame, detections, tracked_objects):
    """
    在帧上绘制检测和跟踪结果
    """
    # 本次观测
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {int(cls)} Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 本次估计
    for obj_id, kf in tracked_objects.items():
        state = kf.x
        x, y = int(state[0]), int(state[1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID: {obj_id}", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

# 模块 5: 主函数
def main(model_path, video_path, conf_threshold, iou_threshold, output_path):
    """
    主函数：加载模型、处理视频、保存结果
    """
    # 加载YOLO模型
    model = YOLO(model_path)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化目标跟踪器
    tracked_objects = {}  # {object_id: kalman_filter}
    next_object_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 目标检测
        detections = detect_objects(model, frame, conf_threshold, iou_threshold)
        
        if len(detections) == 0: 
            if len(tracked_objects) == 0: 
                out.write(frame)
                continue
            else : 
                for obj_id, kf in tracked_objects.items():
                    tracked_objects[obj_id] = update_kalman_filter(kf, None)
        else:
            detection = detections[0] #单目标，其余都是重检
            x1, y1, x2, y2, conf, cls = detection
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32).reshape(2, 1)  # 计算目标中心(列向量)
            if cls not in tracked_objects: 
                tracked_objects[cls] = create_kalman_filter()
            else:
                tracked_objects[cls] = update_kalman_filter(tracked_objects[cls], center)#利用上一次估计和本次观测

        
        # 绘制结果
        frame = draw_results(frame, detections, tracked_objects)
        
        # 写入输出视频
        out.write(frame)
        
        # 显示实时结果
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.waitKey(0) 
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

# 运行主函数
if __name__ == "__main__":
    # 输入参数
    model_path = "D:\\yolov11_prj\\ultralytics-main\\runs\\detect\\train62\\weights\\best.pt"  # YOLO模型路径
    video_path = "D:\\yolov11_prj\\ultralytics-main\\datasets\\kuavo_task_data\\inputvideo.mp4"   # 输入视频路径
    output_path = "D:\\yolov11_prj\\ultralytics-main\\datasets\\kuavo_task_data\\outputvideo.mp4" # 输出视频路径
    conf_threshold = 0.5       # 置信度阈值
    iou_threshold = 0.8       # IOU阈值
    
    # 调用主函数
    main(model_path, video_path, conf_threshold, iou_threshold, output_path)
    
