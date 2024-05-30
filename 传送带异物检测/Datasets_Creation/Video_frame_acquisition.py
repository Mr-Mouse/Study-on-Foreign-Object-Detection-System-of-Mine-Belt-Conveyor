import cv2
import random

# 打开视频文件
video_path = "WIN_20240507_21_56_41_Pro.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 建立一个存储已选帧索引的列表
selected_frames = []

# 随机选择500帧
for _ in range(50):
    # 生成随机帧索引
    random_frame_index = random.randint(0, total_frames - 1)

    # 确保选取的帧索引不重复
    while random_frame_index in selected_frames:
        random_frame_index = random.randint(0, total_frames - 1)

    # 将选取的帧索引添加到列表中
    selected_frames.append(random_frame_index)

# 读取并保存选取的帧
count = 0
while count < total_frames:
    ret, frame = cap.read()

    # 如果当前帧是选取的随机帧，则保存它
    if count in selected_frames:
        # frame = cv2.resize(frame, None, fx=1/6, fy=1/6, interpolation=cv2.INTER_LINEAR)
        # beta = random.randrange(-40, 60, 10)
        frame = cv2.convertScaleAbs(frame[:, 280:720 + 280])
        cv2.imwrite(f"./Foreign_bodies_datasets/images/train/frame_{count}.png", frame)
    count += 1

# 释放视频对象
cap.release()
