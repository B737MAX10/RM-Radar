# coding=utf-8
"""

    @Author : Wenqi Wang
    @file : main.py
    @date : 2024/5/26 上午11:39
    
"""
import threading
import time

import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox

from elements.yolo import YOLO
from myserial import team, serial_port
from arguments import Arguments


opt = Arguments().parse()
print(f"opt: {opt}")


class Detect:
    def __init__(self, source, M):
        self.M = M

        self.cap = cv2.VideoCapture(source)
        print(f"Video source: {source}")
        fps = int(self.cap.get(5))
        fps = fps if fps else 30
        print(f"FPS: {fps}")

        w = int(self.cap.get(3))
        h = int(self.cap.get(4))
        print(f"Video resolution: {w}x{h}")

        self.frame_num = 0

        self.s = h * opt.map_scale / win.t1.map.shape[0]
        self.map = cv2.resize(win.t1.map, (int(win.t1.map.shape[1] * self.s), int(win.t1.map.shape[0] * self.s)))
        print(f"Map resolution: {self.map.shape[1]}x{self.map.shape[0]}")

        self.record = cv2.VideoWriter("record.avi", cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (w, h))

        self.car_detector = YOLO(model_path=opt.car_model_path, conf_thres=opt.car_conf_thres, iou_thres=opt.car_conf_thres, imgsz=opt.car_imgsz, max_det=opt.car_max_det, device=opt.device, classes=opt.car_classes)
        self.armor_detector = YOLO(model_path=opt.armor_model_path, conf_thres=opt.armor_conf_thres, iou_thres=opt.armor_conf_thres, imgsz=opt.armor_imgsz, max_det=opt.armor_max_det, device=opt.device, classes=opt.armor_classes)

        self.IDs = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5, "R7": 7, "B1": 101, "B2": 102, "B3": 103, "B4": 104, "B5": 105, "B7": 107}
        self.n = 0

    def plot_one_box(self, c1, c2, img, color, label, line_thickness):
        # Plots one bounding box on image img
        cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)  # font thickness
        text_size = cv2.getTextSize(label, 0, line_thickness / 3, font_thickness)[0]
        c2 = c1[0] + text_size[0], c1[1] - text_size[1] - 3

        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], font_thickness, cv2.LINE_AA)

    def process_coords(self, x, y):#320
        print(x, y)
        if x in range(2050, 2520) and y in range(1111, 1650):  # 己方环高吊射点
            x -= 300
            y -= 0
        elif x in range(2200, 2600) and y in range(750, 1111):  # 己方环高左
            x -= 270
            y -= 0
        # elif x in range(3100, 4200) and y in range(1370, 2680):#对方环高
        #     x -= 400
        #     y -= 270
        # elif x in range(3950, 4568) and y in range(2250, 3000):#对方吊射点
        #     x -= 450
        #     y -= 245
        elif x in range(4000, 4500) and y in range(930, 1210):  # 对方基地
            x += 200
            y -= 0
        elif x in range(4500, 5200) and y in range(900, 1600):  # 对方基地深
            x += 350
            y -= 0

        print(x, y)

        return int(x * win.t1.scale), int(y * win.t1.scale)

    def detect(self):
        last_send_time = 0
        if team == "R":
            last_send_times = {"B1": 0, "B3": 0, "B4": 0, "B7": 0}
        elif team == "B":
            last_send_times = {"R1": 0, "R3": 0, "R4": 0, "R7": 0}
        try:
            while self.cap.isOpened():
                cost = time.time()

                self.frame_num += 1

                map = self.map.copy()

                ret, frame = self.cap.read()

                if ret:
                    print(f"\nFrame: {self.frame_num}")

                    self.record.write(frame)

                    c = True

                    if win.scale != 1:
                        frame = cv2.resize(frame, (int(frame.shape[1] * win.scale), int(frame.shape[0] * win.scale)), fx=win.scale, fy=win.scale, interpolation=cv2.INTER_LINEAR)

                    pic = frame.copy()

                    # cost0 = time.time() - cost
                    # print(cost0*1000)

                    car_output = self.car_detector.detect(frame)
                    if car_output:
                        for Car in car_output:
                            print(f"Object: {Car}")
                            xyxy = [Car['bbox'][0][0], Car['bbox'][0][1], Car['bbox'][1][0], Car['bbox'][1][1]]
                            x_center = (xyxy[0] + xyxy[2]) / 2
                            y_center = xyxy[3]
                            # The homography matrix is applied to the center of the lower side of the bbox.
                            x = (self.M[0][0] * x_center + self.M[0][1] * y_center + self.M[0][2]) / (self.M[2][0] * x_center + self.M[2][1] * y_center + self.M[2][2])
                            y = (self.M[1][0] * x_center + self.M[1][1] * y_center + self.M[1][2]) / (self.M[2][0] * x_center + self.M[2][1] * y_center + self.M[2][2])
                            x, y = self.process_coords(int(x/win.t1.scale), int(y/win.t1.scale))
                            coords = (int(x*self.s), int(y*self.s))

                            armor_output = self.armor_detector.detect(frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                            if armor_output:
                                for Armor in armor_output:
                                    print(f"Object: {Armor}")
                                    if 'R' in Armor['label']:
                                        self.plot_one_box((Armor['bbox'][0][0] + xyxy[0], Armor['bbox'][0][1] + xyxy[1]), (Armor['bbox'][1][0] + xyxy[0], Armor['bbox'][1][1] + xyxy[1]), pic, (0, 0, 255), Armor['label'], 1)
                                    elif 'B' in Armor['label']:
                                        self.plot_one_box((Armor['bbox'][0][0] + xyxy[0], Armor['bbox'][0][1] + xyxy[1]), (Armor['bbox'][1][0] + xyxy[0], Armor['bbox'][1][1] + xyxy[1]), pic, (255, 0, 0), Armor['label'], 1)
                                if 'R' in Armor['label']:
                                    cv2.circle(map, coords, 5, (0, 0, 255), -1)
                                    self.plot_one_box((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), pic, (0, 0, 255), Armor['label'], 3)
                                elif 'B' in Armor['label']:
                                    cv2.circle(map, coords, 5, (255, 0, 0), -1)
                                    self.plot_one_box((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), pic, (255, 0, 0), Armor['label'], 3)
                                result = {'ID': self.IDs[Armor['label']], 'position': (int(x / win.t1.scale * 5) / 1000, int((3000 - y / win.t1.scale) * 5) / 1000)}
                                print(f"Result: {result}")
                                current_time = time.time()
                                if current_time - last_send_time >= 0.1:
                                    if Armor['label'] in last_send_times and current_time - last_send_times[Armor['label']] >= 0.4:
                                        last_send_times[Armor['label']] = current_time
                                        print(last_send_times)
                                        c = False
                                    # else:
                                    #     if team == "R":
                                    #         result = {'ID': self.IDs['B7'], 'position': (2.237, 0.746)}
                                    #     elif team == "B":
                                    #         result = {'ID': self.IDs['R7'], 'position': (0.592, 0.746)}
                                        serial_port.tx(result)
                                        last_send_time = current_time
                                        print(f"Send: {result}")
                                    self.n += 1
                                    if self.n == 30:
                                        serial_port.tx_double()
                                        self.n = 0
                            else:
                                cv2.circle(map, coords, 5, (0, 255, 0), -1)
                                self.plot_one_box((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), pic, (0, 255, 0), Car['label'], 3)
                                print(f"Result: null(Position:{(int(x / win.t1.scale * 5) / 1000, int((3000 - y / win.t1.scale) * 5) / 1000)})")

                    current_time = time.time()
                    if current_time - last_send_time >= 0.1:
                        if c:
                            if team == "R":
                                result = {'ID': self.IDs['B7'], 'position': (22.5, 7.5)}
                            elif team == "B":
                                result = {'ID': self.IDs['R7'], 'position': (5.5, 7.5)}
                            serial_port.tx(result)
                            last_send_time = current_time
                            print(f"Send: {result}")

                    cost = time.time() - cost if time.time() - cost else 0.0001

                    pic[:map.shape[0], pic.shape[1] - map.shape[1]:] = map
                    cv2.putText(pic, f"FPS:{int(1 / cost)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    cv2.imshow('result', pic)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                    print(
                        "\rProgress: [Frame: %d] [Cost: %.1fms]"
                        % (
                            self.frame_num,
                            cost * 1000
                        )
                    )
        finally:
            self.record.release()

            print(f'\n\nRecord video has been saved!')


class transform0(QWidget):
    def __init__(self):
        super().__init__()

        self.points = []
        self.number = 0

        self.scale = 1

        self.source = opt.camera

        print(f"Video source: {self.source}")
        self.initUI()
        self.show()
        self.setWindowState(Qt.WindowMaximized)

    def initUI(self):
        self.setWindowTitle("Frame")
        # self.resize(2000, 1250)

        layout = QVBoxLayout()
        self.setLayout(layout)

        content = QVBoxLayout()
        self.cap = cv2.VideoCapture(self.source)

        self.picture = QLabel(self)

        self.ok = False

        def update():
            while self.cap.isOpened():
                try:
                    if not self.ok:
                        ret, self.frame = self.cap.read()
                        if ret:
                            if self.frame.shape[1] > opt.size[0] or self.frame.shape[0] > opt.size[1]:
                                self.scale = min(opt.size[0] / self.frame.shape[1], opt.size[1] / self.frame.shape[0])
                                self.frame = cv2.resize(self.frame, (int(self.frame.shape[1] * self.scale), int(self.frame.shape[0] * self.scale)))

                            self.frame0 = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                            self.picture.setPixmap(QPixmap(QImage(self.frame0.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[2] * self.frame.shape[1], QImage.Format_RGB888)))
                        else:
                            self.cap.release()
                except:
                    self.ok = True
                    break

        th = threading.Thread(target=update, daemon=True)
        th.start()

        print(f"Video_scale: {self.scale}")

        content.addWidget(self.picture)
        content.setAlignment(self.picture, Qt.AlignCenter)
        layout.addLayout(content)

        menu = QHBoxLayout()
        self.select = QPushButton("选帧", self)
        self.select.clicked.connect(self.selectBt)
        menu.addWidget(self.select)
        next = QPushButton("继续", self)
        next.clicked.connect(self.nextBt)
        menu.addWidget(next)
        layout.addLayout(menu)

    def mousePressEvent(self, event):
        if self.ok:
            if event.buttons() == Qt.LeftButton:
                x = event.x() - self.picture.x()
                y = event.y() - self.picture.y()
                if 0 <= x <= self.picture.width() and 0 <= y <= self.picture.height():
                    self.points.append([x, y])
                    self.number += 1
                    print(f"Frame_Point: {self.number} ({x},{y})")
                    cv2.circle(self.frame0, (x, y), 5, (255, 255, 255), -1)
                    cv2.putText(self.frame0, str(self.number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    self.picture.setPixmap(QPixmap(QImage(self.frame0.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[2] * self.frame.shape[1], QImage.Format_RGB888)))

    def backBt(self):
        self.close()
        win.setDisabled(False)

    def selectBt(self):
        self.points = []
        self.number = 0

        if not self.ok:
            self.ok = True
            self.select.setText("重选")
        elif self.ok:
            self.ok = False
            self.select.setText("选帧")

    def nextBt(self):
        if len(self.points) < 4:
            QMessageBox.warning(self, "警告", "请至少选择4个点！", QMessageBox.Ok)
            self.points = []
            self.number = 0
            self.frame0 = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.picture.setPixmap(QPixmap(QImage(self.frame0.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[2] * self.frame.shape[1], QImage.Format_RGB888)))
        else:
            self.cap.release()
            self.setDisabled(True)

            win.t1 = transform1()
            win.t1.show()
            win.t1.setWindowState(Qt.WindowMaximized)

    def closeEvent(self, event):
        self.cap.release()
        win.setDisabled(False)


class transform1(QWidget):
    def __init__(self):
        super().__init__()

        self.points = []
        self.number = 0

        self.scale = 1

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Map")
        # self.resize(1250, 750)

        layout = QVBoxLayout()
        self.setLayout(layout)

        content = QVBoxLayout()
        self.picture = QLabel(self)
        self.map = cv2.imread(opt.map)
        if self.map.shape[1] > opt.size[0] or self.map.shape[0] > opt.size[1]:
            self.scale = min(opt.size[0] / self.map.shape[1], opt.size[1] / self.map.shape[0])
            self.map = cv2.resize(self.map, (int(self.map.shape[1] * self.scale), int(self.map.shape[0] * self.scale)))
        self.map0 = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)
        self.picture.setPixmap(QPixmap(QImage(self.map0.data, self.map.shape[1], self.map.shape[0], self.map.shape[2] * self.map.shape[1], QImage.Format_RGB888)))

        print(f"Map_scale: {self.scale}")

        content.addWidget(self.picture)
        content.setAlignment(self.picture, Qt.AlignCenter)
        layout.addLayout(content)

        menu = QHBoxLayout()
        back = QPushButton("返回", self)
        back.clicked.connect(self.backBt)
        menu.addWidget(back)
        next = QPushButton("继续", self)
        next.clicked.connect(self.nextBt)
        menu.addWidget(next)
        layout.addLayout(menu)

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            x = event.x() - self.picture.x()
            y = event.y() - self.picture.y()
            if 0 <= x <= self.picture.width() and 0 <= y <= self.picture.height():
                self.points.append([x, y])
                self.number += 1
                print(f"Map_Point: {self.number} ({x},{y})")
                cv2.circle(self.map0, (x, y), 5, (255, 255, 255), -1)
                cv2.putText(self.map0, str(self.number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                self.picture.setPixmap(QPixmap(QImage(self.map0.data, self.map.shape[1], self.map.shape[0], self.map.shape[2] * self.map.shape[1], QImage.Format_RGB888)))

    def backBt(self):
        self.close()

        win.setDisabled(False)
        win.points = []
        win.number = 0
        win.frame0 = cv2.cvtColor(win.frame, cv2.COLOR_BGR2RGB)
        win.picture.setPixmap(QPixmap(QImage(win.frame0.data, win.frame.shape[1], win.frame.shape[0], win.frame.shape[2] * win.frame.shape[1], QImage.Format_RGB888)))

    def nextBt(self):
        if len(self.points) != len(win.points):
            QMessageBox.warning(self, "警告", "两张图片的点数不一致！", QMessageBox.Ok)
            self.points = []
            self.number = 0
            self.map0 = cv2.cvtColor(self.map, cv2.COLOR_BGR2RGB)
            self.picture.setPixmap(QPixmap(QImage(self.map0.data, self.map.shape[1], self.map.shape[0], self.map.shape[2] * self.map.shape[1], QImage.Format_RGB888)))
        else:
            win.t2 = transform2()
            win.t2.show()
            win.t2.setWindowState(Qt.WindowMaximized)
            self.setDisabled(True)

    def closeEvent(self, event):
        win.setDisabled(False)


class transform2(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Result")
        # self.resize(1250, 750)

        layout = QVBoxLayout()
        self.setLayout(layout)

        content = QVBoxLayout()
        self.picture = QLabel(self)
        srcArr = np.float32(win.points)
        dstArr = np.float32(win.t1.points)
        self.M, status = cv2.findHomography(srcArr, dstArr)
        result = cv2.warpPerspective(win.frame, self.M, (win.t1.map.shape[1], win.t1.map.shape[0]))
        self.picture.setPixmap(QPixmap(QImage(cv2.cvtColor(result, cv2.COLOR_BGR2RGB).data, win.t1.map.shape[1], win.t1.map.shape[0], win.t1.map.shape[2] * win.t1.map.shape[1], QImage.Format_RGB888)))

        content.addWidget(self.picture)
        content.setAlignment(self.picture, Qt.AlignCenter)
        layout.addLayout(content)

        menu = QHBoxLayout()
        back = QPushButton("返回", self)
        back.clicked.connect(self.backBt)
        menu.addWidget(back)
        next = QPushButton("继续", self)
        next.clicked.connect(self.nextBt)
        menu.addWidget(next)
        layout.addLayout(menu)

    def backBt(self):
        self.close()
        win.t1.setDisabled(False)
        win.t1.points = []
        win.t1.number = 0
        win.t1.map0 = cv2.cvtColor(win.t1.map, cv2.COLOR_BGR2RGB)
        win.t1.picture.setPixmap(QPixmap(QImage(win.t1.map0.data, win.t1.map.shape[1], win.t1.map.shape[0], win.t1.map.shape[2] * win.t1.map.shape[1], QImage.Format_RGB888)))

    def nextBt(self):
        win.close()
        win.t1.close()
        self.close()
        app.quit()

        rx = threading.Thread(target=serial_port.rx, daemon=True)
        rx.start()

        detection = Detect(win.source, self.M)
        detection.detect()

    def closeEvent(self, event):
        win.t1.setDisabled(False)


if __name__ == '__main__':
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = transform0()
    sys.exit(app.exec_())
