"""
File: video_test.py
Author: CUMT-Muzihao
Date: 2024/05/13
Description:
"""
import cv2

from ultralytics import YOLO

# 加载模型
model = YOLO(model=r"D:\active file\毕业设计-母子浩\模型代码\模型评估\models\Foreign_bodies\train5\weights"
                   r"\best.pt")
# model.model.names = {0: '锚杆', 1: '矸石', 2: '异物', 3: '木屑'}
# 视频文件
video_path = r"D:\active file\毕业设计-母子浩\模型代码\交互界面\test\异物.mp4"

# 打开视频
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # 获取图像
    res, frame = cap.read()
    # 如果读取成功
    if res:
        # 正向推理
        frame = cv2.resize(frame, [640, 640])
        results = model(frame)  # [:, 280:280 + 720]

        # 绘制结果
        annotated_frame = results[0].plot()

        # 显示图像
        cv2.imshow(winname="YOLOV8", mat=annotated_frame)

        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break

    else:
        break

# 释放链接
cap.release()
# 销毁所有窗口
cv2.destroyAllWindows()
