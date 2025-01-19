import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

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

# 模块 2: 卡尔曼滤波更新
def update_kalman_filter(kf, z): # shape(z) = (2,1) 列向量
    """
    更新卡尔曼滤波器
    """
    # 预测
    kf.predict()
    if z is not None: 
        # 如果有观测结果，更新卡尔曼滤波器
        z = z.reshape(2, 1)
        kf.update(z)  # 使用检测到的位置 [x, y]
    
    return kf  

# 模块 3: 绘制检测和跟踪结果
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

class YOLONode:
    def __init__(self,video_path, output_path, conf_threshold = 0.45, iou_threshold = 0.5):
        self.video_path = video_path
        self.output_path = output_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        rospy.init_node('yolo_node', anonymous=True)
        self.bridge = CvBridge()
        ## if needed
        # self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        ## only using a video as a test
        self.center_pub = rospy.Publisher('src/detected_centers', Point, queue_size=10)

        # 加载 YOLOv11 模型
        self.model = self.load_yolo_model()

        self.process_video()
        
    def load_yolo_model(self):
        model = YOLO("detection/best_cpu.pt")
        rospy.loginfo("YOLOv11 model loaded.")
        return model  # 返回模型对象

    def process_video(self):
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # 初始化目标跟踪器
        tracked_objects = {}  # {object_id: kalman_filter}
        next_object_id = 0
        while True:
            ret, cv_image = cap.read()
            if not ret:
                break
            # detect
            detections = self.detect(cv_image, self.conf_threshold , self.iou_threshold)
            # update the tracked_objects
            if len(detections) == 0: 
                if len(tracked_objects) == 0: 
                    self.center_pub.publish(Point())#if object doesn't exist
                    out.write(cv_image)
                    rospy.loginfo(f"Detected {len(detections)} objects.")
                    rospy.loginfo(f"Published center: ({0}, {0})")
                    continue
                else : #not detected but object exists
                    for obj_id, kf in tracked_objects.items():
                        tracked_objects[obj_id] = update_kalman_filter(kf, None)
            else:#detected
                detection = detections[0] #单目标，其余都是重检
                x1, y1, x2, y2, conf, cls = detection
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32).reshape(2, 1)  # 计算目标中心(列向量)
                if cls not in tracked_objects: 
                    tracked_objects[cls] = create_kalman_filter()
                else:
                    tracked_objects[cls] = update_kalman_filter(tracked_objects[cls], center)#利用上一次估计和本次观测
            # 绘制结果
            cv_image = draw_results(cv_image, detections, tracked_objects)
            # update the msg
            point_msg = Point()
            point_msg.x = tracked_objects[0].x[0]#only one object in this test
            point_msg.y = tracked_objects[0].x[1]
            point_msg.z = 0
            self.center_pub.publish(point_msg)
            # update the log
            rospy.loginfo(f"Detected {len(detections)} objects.")
            rospy.loginfo(f"Published center: ({point_msg.x}, {point_msg.y})")
            # 写入输出视频
            out.write(cv_image)
            
            # 显示实时结果
            cv2.imshow("Tracking", cv_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cv2.waitKey(0)
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output video saved to {self.output_path}")
        
        
    def detect(self, cv_image, conf_threshold, iou_threshold):
        # image->detections
        results = self.model(cv_image, conf = conf_threshold, iou = iou_threshold)
        detections = []
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = box.cls[0].cpu().item()
                detections.append([x1, y1, x2, y2, conf, cls])    
        return detections

    # def image_callback(self, msg):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #         centers = self.process_image(cv_image)

    #         # 发布目标中心点
    #         for center in centers:
    #             point_msg = Point()
    #             point_msg.x = center[0]
    #             point_msg.y = center[1]
    #             point_msg.z = 0  # 2D 图像，z 坐标为 0
    #             self.center_pub.publish(point_msg)
    #             rospy.loginfo(f"Detected center: ({point_msg.x}, {point_msg.y})")
    #     except Exception as e:
    #         rospy.logerr(f"Error in YOLO detection: {e}")

if __name__ == "__main__":
    try:
        node = YOLONode("inputvideo.mp4","outputvideo.mp4")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
