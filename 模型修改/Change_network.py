"""
File: Change_network.py
Author: CUMT-Muzihao
Date: 2024/04/25
Description:修改yolov8文件
"""
import shutil

# 请修改ultralytics软件包绝对路径
ultralytics_path = r'D:/Conda/Pytorch_39/Lib/site-packages/ultralytics/'

modules_path = ultralytics_path + r'nn/modules/'
tasks_path = ultralytics_path + r'nn/'
init_file = '__init__.txt'.replace('.txt', '.py')
tasks_file = 'tasks.txt'.replace('.txt', '.py')

shutil.copy('attention.py', modules_path + 'attention.py')
print("添加attention.py成功")
shutil.copy('block.py', modules_path + 'block.py')
print("修改block.py成功")
shutil.copy('__init__.txt', modules_path + init_file)
print("修改nn.models __init__成功")
shutil.copy('tasks.txt', tasks_path + tasks_file)
print("修改nn.tasks成功")


