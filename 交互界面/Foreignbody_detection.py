# -*- coding:utf-8 -*-
"""
File: Foreignbody_detection.py
Author: CUMT-Muzihao
Date: 2024/05/17
Description:
"""
import os
from threading import Thread
from tkinter import NW, Toplevel, font, ttk, filedialog, Label, Tk, Canvas, BooleanVar
from functools import partial
from cv2 import VideoCapture, CAP_DSHOW, imread, cvtColor, COLOR_BGR2RGB, CAP_PROP_FRAME_COUNT, imencode, \
    CAP_PROP_POS_FRAMES, imdecode, resize, IMREAD_COLOR
import numpy as np
from PIL import Image, ImageTk, ImageDraw

from Predict import YOLO, Result
from ctypes import windll
from datetime import datetime, date


def set_window_size(interface):
    # 指定窗口的宽度和高度
    window_width = 1280
    window_height = 720
    # 获取屏幕的宽度和高度
    screen_width = interface.winfo_screenwidth()
    screen_height = interface.winfo_screenheight()
    # 计算窗口的位置，以便其出现在屏幕中央
    position_x = (screen_width - window_width) // 2
    position_y = (screen_height - window_height) // 2
    # 设置窗口的尺寸和位置
    interface.geometry(f"{window_width}x{window_height}+{position_x}+{position_y - 50}")


def select_path(event, combobox):
    # global path
    path = filedialog.askdirectory()
    combobox.set(path)
    print(path)


def select_file(event, combobox, types):
    file = filedialog.askopenfilename(filetypes=types)
    combobox.set(file)


def show_popup(title):
    popup = Toplevel()
    popup.iconbitmap('./asset/resource/cover.ico')
    # 指定窗口的宽度和高度
    window_width = 200
    window_height = 80
    # 获取屏幕的宽度和高度
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()
    # 计算窗口的位置，以便其出现在屏幕中央
    position_x = (screen_width - window_width) // 2
    position_y = (screen_height - window_height) // 2
    # 设置窗口的尺寸和位置
    popup.geometry(f"{window_width}x{window_height}+{position_x}+{position_y - 50}")
    txt = Label(popup, text=title)
    txt.pack()
    button = ttk.Button(popup, text="确定", command=popup.destroy)
    button.pack()


def list_camera_devices():
    index = 0
    camera_list = []
    while True:
        caps = VideoCapture(index, CAP_DSHOW)
        if not caps.isOpened():
            break
        camera_list.append("摄像头" + str(index))
        caps.release()
        index += 1
    return camera_list


def round_corner_np(image_np, corner_radius, bg_color=None):
    img = Image.fromarray(image_np)
    width, height = img.size
    corner_radius = min(corner_radius, width / 2, height / 2)
    circle = Image.new('L', (corner_radius * 2, corner_radius * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, corner_radius * 2, corner_radius * 2), fill=255)
    alpha = Image.new('L', img.size, 255)
    w, h = img.size
    alpha.paste(circle.crop((0, 0, corner_radius, corner_radius)), (0, 0))
    alpha.paste(circle.crop((0, corner_radius, corner_radius, corner_radius * 2)), (0, h - corner_radius))
    alpha.paste(circle.crop((corner_radius, 0, corner_radius * 2, corner_radius)), (w - corner_radius, 0))
    alpha.paste(circle.crop((corner_radius, corner_radius, corner_radius * 2, corner_radius * 2)),
                (w - corner_radius, h - corner_radius))
    img.putalpha(alpha)
    img = img.convert("RGBA")
    if bg_color:
        bg = Image.new('RGBA', img.size, bg_color + (255,))
        img = Image.alpha_composite(bg, img)
    return img


def image_to_square(image_np, fill_color=None):
    if fill_color is None:
        fill_color = [0, 0, 0]
    height, width, channels = image_np.shape
    if height == width:
        return image_np
    size = max(height, width)
    new_image = np.zeros((size, size, channels), dtype=image_np.dtype)
    # 填充颜色
    new_image[:] = fill_color
    if height > width:
        y_offset = 0
        x_offset = (height - width) // 2
    else:
        y_offset = (width - height) // 2
        x_offset = 0
    new_image[y_offset:y_offset + height, x_offset:x_offset + width] = image_np
    return new_image


last_camera = None
last_video = None
camera_flag = False
image_flag = False
video_flag = False
cap = None
detection_flag = False
last_model = None
Model = None
last_mediatype = None
total_frames = 1
results = Result()


def swich_whlie():
    global detection_flag
    if not detection_flag and checkbutton_var.get():
        if combobox2.get() == '媒体选择' and combobox3.get() == '模型选择':
            checkbutton_var.set(False)
            show_popup("请选择媒体和模型")
        elif combobox2.get() == '媒体选择' and combobox3.get() != '模型选择':
            checkbutton_var.set(False)
            show_popup("请选择媒体")
        elif combobox2.get() != '媒体选择' and combobox3.get() == '模型选择':
            checkbutton_var.set(False)
            show_popup("请选择模型")
        else:
            detection_flag = True
    if not checkbutton_var.get():
        detection_flag = False
    root.after(200, swich_whlie)


def mediatype_while():
    global last_mediatype, cap, camera_flag, last_camera, last_model, Model, image_flag
    global video_flag, last_video, total_frames

    if combobox2.get() == '':
        combobox2.set('媒体选择')
    mediatype = combobox1.get()
    if mediatype == '摄像头' and last_mediatype != mediatype:
        combobox2.unbind("<Button-1>")

        image_ary = imread('./asset/resource/cd.png')
        image_ary = cvtColor(image_ary, COLOR_BGR2RGB)
        img = round_corner_np(image_ary, 15, (255, 255, 255))
        photo = ImageTk.PhotoImage(img)
        canvas.create_image((640 - image_ary.shape[1]) // 2,
                            (640 - image_ary.shape[0]) // 2,
                            anchor=NW, image=photo)
        canvas.image = photo
        root.update()
        combobox2.set("媒体选择")
        combobox2['values'] = list_camera_devices()
        last_mediatype = mediatype
        image_flag = False
        video_flag = False
    if mediatype == '图片' and last_mediatype != mediatype:
        combobox2.set("媒体选择")
        combobox2['values'] = []
        combobox2.bind("<Button-1>",
                       partial(select_file, combobox=combobox2,
                               types=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("ALL", "*.*")]))
        last_mediatype = mediatype
        camera_flag = False
        video_flag = False
    if mediatype == '视频' and last_mediatype != mediatype:
        combobox2.set("媒体选择")
        combobox2['values'] = []
        combobox2.bind("<Button-1>",
                       partial(select_file, combobox=combobox2,
                               types=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("ALL", "*.*")], ))
        last_mediatype = mediatype
        camera_flag = False
        image_flag = False

    camera = combobox2.get()
    if mediatype == '摄像头' and camera != '媒体选择' and last_camera != camera:
        image_ary = imread('./asset/resource/cl.png')
        image_ary = cvtColor(image_ary, COLOR_BGR2RGB)
        img = round_corner_np(image_ary, 15, (255, 255, 255))
        photo = ImageTk.PhotoImage(img)
        canvas.create_image((640 - image_ary.shape[1]) // 2,
                            (640 - image_ary.shape[0]) // 2,
                            anchor=NW, image=photo)
        canvas.image = photo
        root.update()
        last_camera = camera
        cap = None
        cap = VideoCapture(int(last_camera.replace('摄像头', '')))
        last_video = None
        camera_flag = True

    if mediatype == '图片' and combobox2.get() != '媒体选择':
        last_camera = None
        last_video = None
        image_flag = True
    video = combobox2.get()
    if mediatype == '视频' and combobox2.get() != '媒体选择' and last_video != video:
        total_frames = 1
        cap = VideoCapture(video)
        total_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
        last_camera = None
        last_video = video
        video_flag = True

    model = combobox3.get()
    if model != '模型选择' and last_model != model:
        image_ary = imread('./asset/resource/ml.png')
        image_ary = cvtColor(image_ary, COLOR_BGR2RGB)
        img = round_corner_np(image_ary, 15, (255, 255, 255))
        photo = ImageTk.PhotoImage(img)
        canvas.create_image((640 - image_ary.shape[1]) // 2,
                            (640 - image_ary.shape[0]) // 2,
                            anchor=NW, image=photo)
        canvas.image = photo
        root.update()
        Model = YOLO(model_path + model)
        Model(np.random.randint(0, 256, [400, 600, 3], np.uint8))
        last_model = model

    root.after(100, mediatype_while)


def tree_while():
    global results
    file_path = combobox4.get()
    if file_path == '':
        combobox4.set('异物信息保存路径')
        file_path = '异物信息保存路径'
    tree.delete(*tree.get_children())
    if results.data is not None:
        now = datetime.now()
        values = np.array(list(results.obj.names.values()))
        result = np.array(results.data)
        datas = np.empty([result.shape[0], 5], dtype=object)
        datas[:, 0] = values[result[:, 5].astype(int).flatten()].reshape(result.shape[0])
        datas[:, 4] = now.strftime("%H:%M:%S")
        datas[:, 1] = np.char.add((result[:, 4] * 100).round(2).astype(str), '%')
        data_x, data_y = result[:, 0].astype(int), result[:, 1].astype(int)
        data_w, data_h = result[:, 2].astype(int), result[:, 3].astype(int)
        datas[:, 2] = np.array([x.astype(str) + ',' + y.astype(str) for x, y in zip(data_x, data_y)])
        datas[:, 3] = np.array([x.astype(str) + '*' + y.astype(str) for x, y in zip(data_w, data_h)])
        for data in datas.tolist():
            tree.insert("", "end", text='', values=data)

        if file_path != '异物信息保存路径':
            folder_name = str(date.today())
            if not os.path.exists(file_path + '/' + folder_name):
                # 如果不存在，创建文件夹
                os.makedirs(file_path + '/' + folder_name)
            imencode('.jpg', results.plot())[1].tofile(
                file_path + '/' + folder_name + '/' + now.strftime("%H-%M-%S") + '.jpg')
    root.after(1000, tree_while)


# noinspection PyCallingNonCallable
def main_while():
    global camera_flag, image_flag, cap, detection_flag, Model, total_frames, video_flag, results

    if camera_flag:
        _, image_array = cap.read()

    elif image_flag:
        image_array = imdecode(np.fromfile(combobox2.get(), dtype=np.uint8), flags=IMREAD_COLOR)

    elif video_flag and detection_flag:
        res, image_array = cap.read()
        if not res:
            cap.set(CAP_PROP_POS_FRAMES, total_frames - 1)
            _, image_array = cap.read()

    else:
        image_array = imread('./asset/resource/cumt.png')

    image_array = image_to_square(image_array, fill_color=[150, 150, 150])
    image_array = resize(image_array, [640, 640])
    image_array = cvtColor(image_array, COLOR_BGR2RGB)
    if detection_flag:
        results = Model(image_array)
        image_array = results.plot()
    else:
        results = Result()

    img = round_corner_np(image_array, 15, (255, 255, 255))
    photo = ImageTk.PhotoImage(img)
    canvas.create_image((640 - image_array.shape[1]) // 2,
                        (640 - image_array.shape[0]) // 2,
                        anchor=NW, image=photo)
    canvas.image = photo

    if detection_flag:
        root.after(2, main_while)
    else:
        root.after(200, main_while)


if __name__ == "__main__":
    # freeze_support()
    root = Tk()
    windll.shcore.SetProcessDpiAwareness(1)
    ScaleFactor = windll.shcore.GetScaleFactorForDevice(0)
    root.title("皮带机异物检测系统")
    root.iconbitmap('./asset/resource/cover.ico')
    root.call("source", "./asset/azure.tcl")
    root.call("set_theme", "light")
    root.call('tk', 'scaling', ScaleFactor / 100)

    set_window_size(root)

    canvas = Canvas(root, width=640, height=640)  # 设置Canvas大小
    canvas.pack()
    canvas.place(x=40, y=40)

    combobox_font = font.Font(family='微软雅黑', size=15)
    # 定义下拉框选项

    options1 = ["摄像头", "视频", "图片"]
    combobox1 = ttk.Combobox(root, values=options1, font=combobox_font)
    combobox1.option_add('*Listbox.font', combobox_font)
    combobox1.pack(pady=10)
    combobox1.set("媒体类型")
    combobox1.place(x=720, y=140, width=240, height=50)

    options2 = []
    combobox2 = ttk.Combobox(root, values=options2, font=combobox_font)
    combobox2.option_add('*Listbox.font', combobox_font)
    combobox2.pack(pady=10)
    combobox2.set("媒体选择")
    combobox2.place(x=1000, y=140, width=240, height=50)

    model_path = "./asset/models/"
    model_names = os.listdir(model_path)
    options3 = model_names
    combobox3 = ttk.Combobox(root, values=options3, font=combobox_font)
    combobox3.option_add('*Listbox.font', combobox_font)
    combobox3.pack(pady=10)
    combobox3.set("模型选择")
    combobox3.place(x=720, y=230, width=240, height=50)

    combobox4 = ttk.Combobox(root, font=combobox_font)
    combobox4.pack(pady=10)
    combobox4.set("异物信息保存路径")
    combobox4.place(x=1000, y=230, width=240, height=50)
    combobox4.bind("<Button-1>", partial(select_path, combobox=combobox4))

    label = ttk.Label(root, text="开始检测:", font=font.Font(family='微软雅黑', size=20))
    label.place(x=720, y=50)  # 设置标签的位置

    checkbutton_var = BooleanVar()
    checkbox = ttk.Checkbutton(root, style="Switch.TCheckbutton", variable=checkbutton_var)
    checkbox.place(x=880, y=55)  # 设置开关的位置

    columns = ("异物种类", "置信度", "位置", "大小", "时间")
    tree = ttk.Treeview(root, columns=columns, show="headings", )
    tree.pack()
    for col in columns:
        tree.heading(col, text=col)
    tree.place(x=720, y=320, width=520, height=360)
    for col in columns:
        tree.column(col, width=font.Font().measure(col), anchor="center")

    Thread(target=mediatype_while, daemon=True).start()
    swich_whlie()
    main_while()
    tree_while()
    root.mainloop()
