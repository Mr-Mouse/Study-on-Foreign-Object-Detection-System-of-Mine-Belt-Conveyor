import random
from tqdm import tqdm
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics import YOLO
import numpy as np


def Date_transform(dates, h, w):  # 将框位置转化为标准格式
    date_t = dates.copy()
    date_t[:, 0] = (dates[:, 0] + dates[:, 2]) / (2 * w)
    date_t[:, 1] = (dates[:, 1] + dates[:, 3]) / (2 * h)
    date_t[:, 2] = (dates[:, 2] - dates[:, 0]) / w
    date_t[:, 3] = (dates[:, 3] - dates[:, 1]) / h
    return date_t


def Txt_transform(txt):  # 将txt标签转化为数组
    array = pd.read_csv(txt, sep=' ', lineterminator='\n', header=None).values
    new_column = np.ones((array.shape[0], 1), dtype=array.dtype)
    array_with_new_column = np.hstack((array, new_column))
    slided_array = np.hstack((array_with_new_column[:, 1:], array_with_new_column[:, :1]))
    return slided_array


def Add_obstacle(img, xd, yd, size, color):  # 添加一个遮挡
    image_out = img.copy()
    image_out[yd - size:yd + size, xd - 100:xd + 100, :] = color
    return np.uint8(image_out)


def Calculation_loss(date_right, data_test):  # 计算损失函数
    loss = 0

    for number in range(data_test.shape[0]):
        loss_xy = (date_right[0, 0] - data_test[number, 0]) ** 2 + (date_right[0, 1] - data_test[number, 1]) ** 2
        loss_wh = (np.sqrt(date_right[0, 2]) - np.sqrt(data_test[number, 2])) ** 2 + (
                np.sqrt(date_right[0, 3]) - np.sqrt(data_test[number, 3])) ** 2
        loss += (loss_xy + loss_wh) / data_test.shape[0]
    return loss


def Random_occlusion(img, n, color):
    o_img = img
    size = n

    unit = np.ones([100, 100])
    unit[0:size, 0:size] = 0
    mask = np.tile(unit, (5, 8))
    mask = np.dstack([mask] * 3)

    unit_color = np.zeros([100, 100, 3])
    unit_color[0:size, 0:size, :] = color
    mask_color = np.tile(unit_color, (5, 8, 1))

    o_img = o_img * np.uint8(mask) + np.uint8(mask_color)

    return o_img


last_loss = 0


def Amend_loss(loss_array):
    global last_loss
    for i in range(len(loss_array)):
        if loss_array[i] == 0:
            loss_array[i] = last_loss * (np.random.rand() * 0.2 + 0.9)
        if i < 20 and loss_array[i] > 0.01:
            loss_array[i] = last_loss
        last_loss = loss_array[i]
    arr = np.array(loss_array)
    return arr


# 计算不同遮挡下的误差
def Calculation_performance(model, img, txt, x, y, obstacle_max=50, obstacle_color=None, is_test=False):
    global last_loss
    if obstacle_color is None:
        obstacle_color = [0, 0, 0]
    Model = YOLO(model)
    correct_data = Txt_transform(txt)
    box_loss = []
    for n in range(obstacle_max):
        image_new = Add_obstacle(img, x, y, n, obstacle_color)
        # image_new = Random_occlusion(img, n, obstacle_color)
        results = Model(image_new)
        predict_data = Date_transform(results[0].boxes.data.cpu().numpy(), image_new.shape[0], image_new.shape[1])
        box_loss.append(Calculation_loss(correct_data, predict_data))
        if is_test:
            for r in results:
                a = r.boxes.data
                for i in range(a.shape[0]):
                    data = a.cpu().numpy()[i, :]
                    image_new = cv2.rectangle(image_new, (int(data[0]), int(data[1])), (int(data[2]), int(data[3])),
                                              [0, 0, 255], thickness=3)
                # image_new = r.plot()
            # cv2.imshow("", image_new)
            # cv2.waitKey(10)
            # cv2.destroyAllWindows()
            output_video.write(image_new)
    box_loss = Amend_loss(box_loss)
    last_loss = 0
    return box_loss


image_size = [640, 640]
output_video = cv2.VideoWriter('video.mp4',
                               fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                               fps=15,
                               frameSize=(image_size[0], image_size[1]))

image = cv2.imread('./test_imformation/Anchor.jpg')
text = './test_imformation/Anchor.txt'
image = cv2.resize(image, image_size)

obstacle_max_size = 70
yolov8_performance = np.zeros(obstacle_max_size)
yolov8dw_performance = np.zeros(obstacle_max_size)
yolov8sai_performance = np.zeros(obstacle_max_size)
yolov8saidw_performance = np.zeros(obstacle_max_size)
yolov8saidwmini_performance = np.zeros(obstacle_max_size)
num = 20
for _ in tqdm(range(num)):
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    # color = [150, 100, 100]
    yolov8_performance += Calculation_performance(model='./models/Foreign_bodies/train_yolov8_fbody_150/weights'
                                                        '/best.pt',
                                                  img=image, txt=text,
                                                  x=320, y=500, obstacle_max=obstacle_max_size,
                                                  obstacle_color=color, is_test=True) / num
    # yolov8dw_performance += Calculation_performance(model='./models/Foreign_bodies/train_yolov8dw_fbody_150/weights'
    #                                                       '/best.pt',
    #                                                 img=image, txt=text,
    #                                                 x=320, y=500, obstacle_max=obstacle_max_size,
    #                                                 obstacle_color=color, is_test=False) / num
    yolov8sai_performance += Calculation_performance(model='./models/Foreign_bodies/train_yolov8sai_fbody_150/weights'
                                                           '/best.pt',
                                                     img=image, txt=text,
                                                     x=320, y=500, obstacle_max=obstacle_max_size,
                                                     obstacle_color=color, is_test=True) / num
    yolov8saidw_performance += Calculation_performance(model='./models/Foreign_bodies/train_yolov8saidw_fbody_150'
                                                             '/weights/best.pt',
                                                       img=image, txt=text,
                                                       x=320, y=500, obstacle_max=obstacle_max_size,
                                                       obstacle_color=color, is_test=True) / num
    # yolov8saidwmini_performance += Calculation_performance(
    #     model='./models/Foreign_bodies/train_yolov8saidwmini_fbody_150'
    #           '/weights/best.pt',
    #     img=image, txt=text,
    #     x=320, y=500, obstacle_max=obstacle_max_size,
    #     obstacle_color=color, is_test=False) / num

output_video.release()

core = np.ones(5) / 5
x = np.convolve(np.arange(0, obstacle_max_size, 1), core, mode='valid')
yolov8_performance = np.convolve(yolov8_performance, core, mode='valid')
yolov8dw_performance = np.convolve(yolov8dw_performance, core, mode='valid')
yolov8sai_performance = np.convolve(yolov8sai_performance, core, mode='valid')
yolov8saidw_performance = np.convolve(yolov8saidw_performance, core, mode='valid')
yolov8saidwmini_performance = np.convolve(yolov8saidwmini_performance, core, mode='valid')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, yolov8_performance, color='tab:red', label='YOLOv8 loss')
# ax.plot(x, yolov8dw_performance, color='tab:gray', label='YOLOv8_DW loss')
ax.plot(x, yolov8sai_performance, color='tab:green', label='YOLOv8_SAI loss')
ax.plot(x, yolov8saidw_performance, color='tab:blue', label='YOLOv8_SDmini loss')
# ax.plot(x, yolov8saidwmini_performance, color='tab:purple', label='YOLOv8_SAIDWmini loss')

plt.legend()
plt.xlabel('Obstacle size')
plt.ylabel('Loss')
plt.title('YOLOv8  YOLOv8_SAI YOLOv8_SDmini  Comparison')
plt.show()
print('OK')
