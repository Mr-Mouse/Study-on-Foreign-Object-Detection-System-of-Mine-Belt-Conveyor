"""
File: mAP_assess.py
Author: CUMT-Muzihao
Date: 2024/05/11
Description:
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def draw_line(scene, path, line_name='', x_name='epoch', y_name='metrics/mAP50(B)', color=None):
    if color is None:
        color = [0, 0, 255]
    csv = pd.read_csv(path)
    csv.columns = csv.columns.str.strip()
    x = csv[x_name].values
    y = csv[y_name].values
    core = np.ones(10)/10
    x = np.convolve(x, core, mode='valid')
    y = np.convolve(y, core, mode='valid')

    scene.plot(x[:100], y[:100], color=tuple(c / 255 for c in color), label=line_name)
    print(line_name+f":max={np.max(y)}")


YOLOv8_csv_path = r"./models\Foreign_bodies\train_yolov8_fbody_150\results.csv"
YOLOv8DW_csv_path = r"./models\Foreign_bodies\train_yolov8dw_fbody_150\results.csv"
# YOLOv8SA_csv_path = r"./models\cocoe\train_yolov8sa_cocoe_150\results.csv"
YOLOv8SAI_csv_path = r"./models\Foreign_bodies\train_yolov8sai_fbody_150\results.csv"
YOLOv8SAIDW_csv_path = r"./models\Foreign_bodies\train_yolov8saidw_fbody_150" \
                       r"\results.csv"
YOLOv8SAIDWmini_csv_path = r"./models\Foreign_bodies\train_yolov8saidwmini_fbody_150" \
                       r"\results.csv"
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xname = 'epoch'
yname = 'metrics/mAP50(B)'
draw_line(ax, YOLOv8_csv_path, 'YOLOv8', x_name=xname, y_name=yname, color=[0, 0, 255])
# draw_line(ax, YOLOv8SA_csv_path, 'YOLOv8_SA', x_name=xname, y_name=yname, color=[255, 0, 0])
draw_line(ax, YOLOv8SAI_csv_path, 'YOLOv8_SAI', x_name=xname, y_name=yname, color=[0, 255, 0])
draw_line(ax, YOLOv8DW_csv_path, 'YOLOv8DW', x_name=xname, y_name=yname, color=[0, 255, 255])
draw_line(ax, YOLOv8SAIDW_csv_path, 'YOLOv8_SAIDW', x_name=xname, y_name=yname, color=[255, 0, 255])
draw_line(ax, YOLOv8SAIDWmini_csv_path, 'YOLOv8_SDmini', x_name=xname, y_name=yname, color=[255, 0, 0])
plt.legend()
plt.xlabel(xname)
plt.ylabel(yname.replace('metrics/', ''))
plt.title("Foreign_bodies "+yname.replace('metrics/', ''))
plt.savefig("Foreign_bodies "+yname.replace('metrics/', ''))
plt.show()

