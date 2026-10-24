红外图像化工厂区火情检测数据集，11662张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11b98a9223af4db3bf7b7d74cac5498b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/416f3f5f6fdf4f90804109ff2c8ab441.png)

图像尺寸:512*512
类别数量:5类
训练集图像数量:10600; 验证集图像数量:910； 测试集图像数量:152
类别名称: 每一类图像数 ，每一类标注数
flame-火焰：6986,7017
fire smoke-火灾烟雾：6707,6745
flare stack-火炬烟囱：6015,14345
normal smoke-普通烟气：6922,12611
chimney-烟囱：8400,17806
image num: 11662

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d9a6c7489833401abc48aa481f8e1c0a.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b3f3e79790641a881e6dbd5f9e9df47.png)


模型代码
采用 YOLOv11n 网络训练![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/272ececf5e644c5587d9d28578b256de.png)

训练轮次：80 个 epoch
提供全部训练 + 测试源代码![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/380c10e235074a0689a40f7c6fc293df.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4c5f81817cae409c9bd3a8f978b43576.png)

训练精度 mAP 效果如图所示
PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/331d9dbeb0744ac3a2ee7e0e5518d5c5.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测
界面实时显示：目标位置、目标总数、置信度等信息
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/69babd7f54b645eab81e8e06c7713481.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2896e1a075bf4532985e39a97b84e929.png)

4、
运行环境：Python=3.8、opencv-python、PyQt5、torch
支持 Windows、Linux 系统


# 基于YOLOv11的红外工厂火情检测系统完整代码
> 功能：图片检测、视频检测、摄像头实时推理、结果保存、PyQt5可视化界面；类别：`flame(火焰)`、`fire smoke(火情烟)`、`flare stack(火炬塔)`、`normal smoke(正常烟气)`、`chimney(烟囱)`
## 1. 环境安装
```bash
pip install ultralytics pyqt5 opencv-python numpy pillow
```
## 2. 训练脚本 train.py
```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data="fire_thermal.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        patience=10,
        save=True
    )
```
`fire_thermal.yaml`
```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test
names:
  0: flame
  1: fire smoke
  2: flare stack
  3: normal smoke
  4: chimney
```
## 3. PyQt主界面代码 main.py
```python
import sys
import cv2
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QComboBox, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

class DetectThread(QThread):
    result_signal = pyqtSignal(dict)
    def __init__(self, model, src, mode="img"):
        super().__init__()
        self.model = model
        self.src = src
        self.mode = mode
        self.run_flag = True
    def run(self):
        if self.mode == "img":
            res = self.model.predict(self.src, conf=0.5)[0]
            img_plot = res.plot()
            boxes_info = []
            for b in res.boxes:
                cls = self.model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                xyxy = [int(x) for x in b.xyxy[0]]
                boxes_info.append({"cls":cls,"conf":conf,"xyxy":xyxy})
            self.result_signal.emit({"img":img_plot,"boxes":boxes_info})
        elif self.mode == "video":
            cap = cv2.VideoCapture(self.src)
            while self.run_flag and cap.isOpened():
                ret,frame = cap.read()
                if not ret:break
                res = self.model.predict(frame,conf=0.5)[0]
                img_plot = res.plot()
                boxes_info = []
                for b in res.boxes:
                    cls = self.model.names[int(b.cls[0])]
                    conf = float(b.conf[0])
                    xyxy = [int(x) for x in b.xyxy[0]]
                    boxes_info.append({"cls":cls,"conf":conf,"xyxy":xyxy})
                self.result_signal.emit({"img":img_plot,"boxes":boxes_info})
            cap.release()
        elif self.mode == "cam":
            cap = cv2.VideoCapture(0)
            while self.run_flag and cap.isOpened():
                ret,frame = cap.read()
                if not ret:break
                res = self.model.predict(frame,conf=0.5)[0]
                img_plot = res.plot()
                boxes_info = []
                for b in res.boxes:
                    cls = self.model.names[int(b.cls[0])]
                    conf = float(b.conf[0])
                    xyxy = [int(x) for x in b.xyxy[0]]
                    boxes_info.append({"cls":cls,"conf":conf,"xyxy":xyxy})
                self.result_signal.emit({"img":img_plot,"boxes":boxes_info})
            cap.release()
    def stop(self):
        self.run_flag=False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的红外图像工厂火情检测")
        self.resize(1250,860)
        self.model = YOLO("best.pt")
        self.thread = None
        self.init_ui()
    def init_ui(self):
        # 图像显示label
        self.label_img = QLabel(self)
        self.label_img.setGeometry(20,80,600,500)
        self.label_img.setStyleSheet("border:1px solid #999")
        # 文件导入按钮
        self.btn_img = QPushButton("选择图片",self)
        self.btn_img.setGeometry(660,80,160,40)
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video = QPushButton("选择视频文件",self)
        self.btn_video.setGeometry(660,130,160,40)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_cam = QPushButton("开启摄像头",self)
        self.btn_cam.setGeometry(660,180,160,40)
        self.btn_cam.clicked.connect(self.open_cam)
        # 检测结果面板
        self.label_info = QLabel("检测结果",self)
        self.label_info.setGeometry(660,240,200,30)
        self.combo_target = QComboBox(self)
        self.combo_target.addItems(["全部"])
        self.combo_target.setGeometry(660,280,160,30)
        self.label_conf = QLabel("置信度：--",self)
        self.label_conf.setGeometry(660,330,200,30)
        self.label_pos = QLabel("目标位置：",self)
        self.label_pos.setGeometry(660,370,220,100)
        # 表格
        self.table = QTableWidget(self)
        self.table.setGeometry(20,600,600,220)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])
        # 操作按钮
        self.btn_save = QPushButton("保存结果",self)
        self.btn_save.setGeometry(660,520,120,40)
        self.btn_exit = QPushButton("退出",self)
        self.btn_exit.setGeometry(800,520,120,40)
        self.btn_exit.clicked.connect(self.close)
    def show_result(self,ret):
        img = ret["img"]
        boxes = ret["boxes"]
        # 转Qt图像
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = rgb.shape
        qimg = QImage(rgb.data,w,h,w*c,QImage.Format_RGB888)
        self.label_img.setPixmap(QPixmap.fromImage(qimg).scaled(self.label_img.size(),Qt.KeepAspectRatio))
        # 填充表格
        self.table.setRowCount(len(boxes))
        for i,bx in enumerate(boxes):
            self.table.setItem(i,0,QTableWidgetItem(str(i+1)))
            self.table.setItem(i,1,QTableWidgetItem("current file"))
            self.table.setItem(i,2,QTableWidgetItem(bx["cls"]))
            self.table.setItem(i,3,QTableWidgetItem(f"{bx['conf']:.2%}"))
            self.table.setItem(i,4,QTableWidgetItem(str(bx["xyxy"])))
        if boxes:
            self.label_conf.setText(f"置信度 {boxes[0]['conf']:.2%}")
            self.label_pos.setText(f"xmin:{boxes[0]['xyxy'][0]} ymin:{boxes[0]['xyxy'][1]}\nxmax:{boxes[0]['xyxy'][2]} ymax:{boxes[0]['xyxy'][3]}")
    def load_img(self):
        fpath,_ = QFileDialog.getOpenFileName(self,"打开图片",filter="图片(*.jpg *.png *.jpeg)")
        if not fpath:return
        self.thread = DetectThread(self.model,fpath,mode="img")
        self.thread.result_signal.connect(self.show_result)
        self.thread.start()
    def load_video(self):
        fpath,_ = QFileDialog.getOpenFileName(self,"打开视频",filter="视频(*.mp4 *.avi)")
        if not fpath:return
        if self.thread:self.thread.stop()
        self.thread = DetectThread(self.model,fpath,mode="video")
        self.thread.result_signal.connect(self.show_result)
        self.thread.start()
    def open_cam(self):
        if self.thread:self.thread.stop()
        self.thread = DetectThread(self.model,0,mode="cam")
        self.thread.result_signal.connect(self.show_result)
        self.thread.start()
    def closeEvent(self,event):
        if self.thread:self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
```
## 4. 推理单图脚本 predict.py
```python
from ultralytics import YOLO
model = YOLO("best.pt")
res = model.predict("test.jpg",save=True,conf=0.5)
for r in res:
    for box in r.boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()
        print(f"类别:{cls_name},置信度:{conf:.3f},坐标:{xyxy}")
```
## 使用说明
1. 将训练完成权重 `best.pt` 和代码放同一目录
2. 数据集按YOLO格式整理，5类标签和yaml匹配
3. 运行`main.py`直接唤起QT界面，支持图片、视频、摄像头红外火情检测，输出火焰、火情烟雾、火炬塔、正常烟气、烟囱检测框+置信度+坐标
4. 可直接用于工业厂区红外监控、无人机红外巡检火情预警

？
