import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QFileDialog,
                             QPushButton, QSlider, QHBoxLayout, QGroupBox, QLineEdit)
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import os


class MotionDetector:
    def __init__(self, min_area=1000, sub_threshold=25, distance_threshold=200):
        '''
        Docstring для MotionDetector

        :param min_area: параметр отвечающий за минимальный размер bbox маски
        :param sub_threshold: порог дисперсии, пиксели с дисперсией ниже этого порога считаются фоновыми.
        :param distance_threshold: расстояние для кластеризации
        '''
        # TODO: сделать автоматическая калибровка при использовании флажка
        self.min_area = min_area
        self.sub_threshold = sub_threshold
        self.distance_threshold = distance_threshold

        # Инициализация детекторов
        self.update_subtractor()
        self.update_cluster()

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # ядро для применения поиска

    def update_subtractor(self):
        """Обновляет фоновый вычитатель"""
        # Создание фонового вычитателя
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=100,  # длина истории
            varThreshold=self.sub_threshold,  # порог дисперсии
            detectShadows=True  # обнаружение теней
        )

    def update_cluster(self):
        """Обновляет кластеризатор"""
        # Создание кластеризатора
        self.cluster = DBSCAN(eps=self.distance_threshold, min_samples=1)

    def detect(self, frame):
        '''
        Docstring для detect

        :param frame: отдельный кадр
        '''
        # Фоновая субтракция
        fg_mask = self.sub.apply(frame)

        # Удаление шума
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)  # открытие
        # Заполнение отверстий
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)  # закрытие
        # Усреднение значений пикселей
        fg_mask = cv2.medianBlur(fg_mask, 5)  # медианная фильтрация

        # Поиск контура
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes = []
        for contour in contours:
            # Площадь многоугольника ограниченного контуром
            area = cv2.contourArea(contour)
            # Отсеивание по размеру
            if area > self.min_area:
                # Поиск крайних точек TODO: улучшить алгоритм поиска
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))

        if len(motion_boxes) == 0:
            return [], fg_mask

        centers = [(x + w / 2, y + h / 2) for x, y, w, h in motion_boxes]

        # Кластеризация по центрам
        clusters = self.cluster.fit(centers)

        merged_boxes = []
        for label in set(clusters.labels_):
            # Объединение bboxes
            clustered_boxes = np.array(motion_boxes)[clusters.labels_ == label]
            x_min = clustered_boxes[:, 0].min()
            y_min = clustered_boxes[:, 1].min()
            x_max = (clustered_boxes[:, 0] + clustered_boxes[:, 2]).max()
            y_max = (clustered_boxes[:, 1] + clustered_boxes[:, 3]).max()
            merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return merged_boxes, fg_mask


class VideoDisplay(QLabel):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        # Настройка отображения видео
        self.setText(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("border: 1px solid gray;")

    def update_frame(self, cv_image):
        """Основной метод: обновляет виджет новым кадром из OpenCV."""
        if cv_image is None:
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.setPixmap(scaled_pixmap)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_camera()
        self.save_interval = 5
        self.record = False
        self.video_writer = None

    def init_ui(self):
        self.setWindowTitle("")
        self.setGeometry(100, 100, 900, 500)

        # Главный layout
        main_layout = QHBoxLayout()

        # Левая часть для видео
        video_layout = QVBoxLayout()

        # видео окна
        video_row = QHBoxLayout()
        self.video_display = VideoDisplay("Основное видео")
        self.mask_display = VideoDisplay("Маска движения")
        video_row.addWidget(self.video_display)
        video_row.addWidget(self.mask_display)

        # Счетчик под видео
        # TODO: добавить консоль с сообщениями о записи, детекциях и ошибках
        stats_layout = QHBoxLayout()
        self.detections_label = QLabel("Детекций: 0")
        stats_layout.addWidget(self.detections_label)

        video_layout.addLayout(video_row)
        video_layout.addLayout(stats_layout)

        # Правая часть - управление
        control_layout = QVBoxLayout()

        # Кнопки
        self.start_btn = QPushButton("Старт")
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setEnabled(False)

        self.record_btn = QPushButton("Запись")
        self.stop_record_btn = QPushButton("Остановить запись")
        self.stop_record_btn.setEnabled(False)

        params_group = QGroupBox("Параметры детектора")
        params_layout = QVBoxLayout()

        self.area_slider = QSlider(Qt.Horizontal)
        self.area_slider.setRange(100, 5000)
        self.area_slider.setValue(1000)
        self.area_label = QLabel(f"Порог площади: {self.area_slider.value()}")
        params_layout.addWidget(self.area_label)
        params_layout.addWidget(self.area_slider)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(5, 100)
        self.threshold_slider.setValue(25)
        self.threshold_label = QLabel(f"Порог дисперсии: {self.threshold_slider.value()}")
        params_layout.addWidget(self.threshold_label)
        params_layout.addWidget(self.threshold_slider)

        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(50, 500)
        self.distance_slider.setValue(200)
        self.distance_label = QLabel(f"Расстояние кластеризации: {self.distance_slider.value()}")
        params_layout.addWidget(self.distance_label)
        params_layout.addWidget(self.distance_slider)

        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 15)  # нечетные числа
        self.kernel_slider.setValue(5)
        self.kernel_label = QLabel(f"Размер ядра: {self.kernel_slider.value()}")
        params_layout.addWidget(self.kernel_label)
        params_layout.addWidget(self.kernel_slider)

        # TODO: выбор устройства ввода
        logi_set = QHBoxLayout()
        default_folder = os.path.join(os.path.abspath(os.curdir), "logs")
        self.path = QLineEdit(default_folder)
        self.path.setReadOnly(True)
        self.path_btn = QPushButton("Обзор")
        logi_set.addWidget(self.path_btn)
        logi_set.addWidget(self.path)
        params_layout.addLayout(logi_set)
        self.path_btn.clicked.connect(self.browse_folder)

        params_group.setLayout(params_layout)

        # Сборка правой панели
        # TODO: добавить кнопку выход
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.stop_record_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(params_group)
        control_layout.addStretch()

        # Сборка главного окна
        main_layout.addLayout(video_layout, 70)
        main_layout.addLayout(control_layout, 30)

        self.setLayout(main_layout)

        # Создаем детектор с начальными значениями
        self.detector = MotionDetector(
            min_area=self.area_slider.value(),
            sub_threshold=self.threshold_slider.value(),
            distance_threshold=self.distance_slider.value()
        )

        self.update_kernel_size()

        # Подключение кнопок
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        self.record_btn.clicked.connect(self.start_recording)
        self.stop_record_btn.clicked.connect(self.stop_recording)

        self.area_slider.valueChanged.connect(self.on_area_change)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        self.distance_slider.valueChanged.connect(self.on_distance_change)
        self.kernel_slider.valueChanged.connect(self.on_kernel_change)

    def start_recording(self):
        if not self.timer.isActive():
            return

        folder_path = self.path.text()
        if not folder_path:
            folder_path = os.path.join(os.path.abspath(os.curdir), "logs")
            self.path.setText(folder_path)

        os.makedirs(folder_path, exist_ok=True)

        ret, frame = self.cap.read()
        if not ret:
            print("Не удалось получить кадр для инициализации записи")
            return

        # Генерируем имя файла с временной меткой
        timestamp = datetime.now()
        filename = f"recording_{timestamp}".replace(":", "-").replace(".", "-") + ".mp4"
        filepath = os.path.join(folder_path, filename)

        frame_size = (frame.shape[1], frame.shape[0])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0

        self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

        if self.video_writer.isOpened():
            self.record = True
            self.record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(True)

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.record = False
        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "")
        if folder:
            self.path.setText(folder)

    def save_frame(self, frame, motion_boxes):
        current_time = datetime.now()
        time_diff = (current_time - self.last_save_time).total_seconds()
        if time_diff < self.save_interval:
            return

        if len(motion_boxes) > 0:
            save_frame = frame.copy()

            save_folder = self.path.text()

            if not save_folder:
                save_folder = os.path.join(os.path.abspath(os.curdir), "recordings")
                self.path.setText(save_folder)

            os.makedirs(save_folder, exist_ok=True)

            filename = f"{current_time}".replace(":", "-").replace(".", "-") + ".jpg"
            filepath = os.path.join(save_folder, filename)
            cv2.imwrite(filepath, save_frame)

            self.last_save_time = current_time

    def update_kernel_size(self):
        value = self.kernel_slider.value()
        if value % 2 == 0:
            kernel_size = value + 1  # делаем нечетным
        else:
            kernel_size = value
        self.detector.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.kernel_label.setText(f"Размер ядра: {kernel_size}")

    def on_area_change(self, value):
        self.detector.min_area = value
        self.area_label.setText(f"Порог площади: {value}")

    def on_threshold_change(self, value):
        self.detector.sub_threshold = value
        self.detector.update_subtractor()
        self.threshold_label.setText(f"Порог дисперсии: {value}")

    def on_distance_change(self, value):
        self.detector.distance_threshold = value
        self.detector.update_cluster()
        self.distance_label.setText(f"Расстояние кластеризации: {value}")

    def on_kernel_change(self, value):
        self.update_kernel_size()

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть камеру.")
            self.video_display.setText("Не удалось открыть камеру")
            self.mask_display.setText("Не удалось открыть камеру")
            return

        # Установка разрешения
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Создание таймера для обновления кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def start_capture(self):
        # Проверяем, открыта ли камера
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("Камера не открыта")
            return

        self.last_save_time = datetime.now() - timedelta(seconds=(self.save_interval - 2))
        if not self.timer.isActive():
            self.timer.start(30)  # ~33 FPS
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Не удалось получить кадр")
            self.timer.stop()
            return

        # Детекция движения
        motion_boxes, mask = self.detector.detect(frame)

        # Рисуем bboxes
        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        display_frame = frame.copy()
        if self.record:
            cv2.putText(display_frame, "REC", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # трехканальное изображение для отображения
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # ОБНОВЛЯЕМ ВИДЖЕТЫ, а не отдельные окна
        self.video_display.update_frame(display_frame)
        self.mask_display.update_frame(mask_display)

        # Обновляем счетчик
        self.detections_label.setText(f"Детекций: {len(motion_boxes)}")

        self.save_frame(frame, motion_boxes)

        if self.record and self.video_writer is not None:
            self.video_writer.write(frame)

    def stop_capture(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        if self.record:
            self.stop_recording()

    def closeEvent(self, event):
        self.stop_capture()
        if hasattr(self, 'cap'):
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())