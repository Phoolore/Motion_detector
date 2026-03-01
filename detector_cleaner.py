import sys
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"
import cv2
import logging
import numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QTextCursor, QFont
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QFileDialog, QStyle,
    QPushButton, QSlider, QHBoxLayout, QGroupBox, QLineEdit, QPlainTextEdit,
    QComboBox, QFrame
)
from sklearn.cluster import DBSCAN
from datetime import datetime
from pathlib import Path
from collections import deque
import threading
import time
import stat
from contextlib import contextmanager


class ManagedVideoWriter:
    """
    Класс для управления видеозаписью с автоматической очисткой хранилища.
    При создании нового видеофайла и его последующем закрытии проверяется
    общий размер всех .mp4 файлов в целевой папке. Если суммарный размер
    превышает заданный лимит (storage_limit_mb), самые старые файлы удаляются.
    """

    def __init__(self, filename, fourcc, fps, frame_size, storage_limit_mb=10):
        """
        :param filename: полный путь к файлу для записи
        :param fourcc: кодек (например, cv2.VideoWriter_fourcc(*'mp4v'))
        :param fps: частота кадров
        :param frame_size: размер кадра (width, height)
        :param storage_limit_mb: максимальный суммарный размер всех видеофайлов в папке (МБ)
        """
        self.filename = filename
        self.storage_limit_mb = storage_limit_mb
        self.folder = os.path.dirname(filename)

        # Перед созданием нового файла попытаемся освободить место
        self._enforce_storage_limit(before_write=True)

        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    def isOpened(self):
        """Возвращает True, если внутренний writer успешно открыт."""
        return self.writer is not None and self.writer.isOpened()

    def write(self, frame):
        """Записывает очередной кадр в видеофайл."""
        if self.writer:
            self.writer.write(frame)

    def release(self):
        """Закрывает видеофайл и запускает проверку лимита хранилища."""
        if self.writer:
            self.writer.release()
            self.writer = None
            self._enforce_storage_limit()

    def _enforce_storage_limit(self, before_write=False):
        """"Проверка и очистка хранилища"""
        base_folder = Path(self.folder).parent if "events" in self.folder else Path(self.folder)
        if not base_folder.exists():
            return

        files = []
        total_size = 0

        # Рекурсивный поиск
        for path in base_folder.rglob("*.mp4"):
            try:
                if before_write and str(path) == self.filename:
                    continue
                size = path.stat().st_size
                mtime = path.stat().st_mtime
                files.append((mtime, path, size))
                total_size += size
            except Exception:
                continue

        limit_bytes = self.storage_limit_mb * 1024 * 1024
        if total_size <= limit_bytes:
            return

        # Сортируем от старых к новым
        files.sort(key=lambda x: x[0])

        # Удаление
        deleted_count = 0
        for mtime, path, size in files:
            if total_size <= limit_bytes:
                break
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if sys.platform == 'win32':
                        path.chmod(stat.S_IWRITE)
                    if attempt > 0:
                        time.sleep(0.5 * attempt)
                    path.unlink()
                    total_size -= size
                    deleted_count += 1
                    break 
                except PermissionError:
                    continue
                except Exception:
                    break


class CircularBuffer:
    """Кольцевой буфер для хранения кадров до события"""
    def __init__(self, max_frames):
        self.buffer = deque(maxlen=max_frames)
        self.lock = threading.Lock()

    def add(self, frame):
        with self.lock:
            self.buffer.append(frame.copy())

    def get_all(self):
        with self.lock:
            return list(self.buffer)

    def clear(self):
        with self.lock:
            self.buffer.clear()


class EventRecorder:
    """Запись событий (клипов) по движению"""
    def __init__(self, fps, post_seconds=5.0):
        self.fps = fps
        self.post_frames = int(post_seconds * fps)
        self.is_recording = False
        self.frames = []
        self.last_motion_time = None
        self.event_time = None

    def start(self, event_time, pre_frames):
        self.is_recording = True
        self.frames = pre_frames.copy()
        self.event_time = event_time
        self.last_motion_time = event_time

    def add_motion_frame(self, frame, current_time):
        self.frames.append(frame.copy())
        self.last_motion_time = current_time

    def add_idle_frame(self, frame, current_time):
        self.frames.append(frame.copy())
        idle_time = (current_time - self.last_motion_time).total_seconds()
        return idle_time >= (self.post_frames / self.fps)

    def stop(self):
        self.is_recording = False
        self.frames = []


class MotionDetector:
    """Детектор движения на основе фоновой субтракции и кластеризации"""
    def __init__(self, min_area=1000, sub_threshold=25, distance_threshold=200):
        self.min_area = min_area
        self.sub_threshold = sub_threshold
        self.distance_threshold = distance_threshold
        self.update_subtractor()
        self.update_cluster()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update_subtractor(self):
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=self.sub_threshold,
            detectShadows=True
        )

    def update_cluster(self):
        self.cluster = DBSCAN(eps=self.distance_threshold, min_samples=1)

    def detect(self, frame):
        fg_mask = self.sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.medianBlur(fg_mask, 5)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))

        if len(motion_boxes) == 0:
            return [], fg_mask

        centers = [(x + w / 2, y + h / 2) for x, y, w, h in motion_boxes]
        clusters = self.cluster.fit(centers)

        merged_boxes = []
        for label in set(clusters.labels_):
            clustered_boxes = np.array(motion_boxes)[clusters.labels_ == label]
            x_min = clustered_boxes[:, 0].min()
            y_min = clustered_boxes[:, 1].min()
            x_max = (clustered_boxes[:, 0] + clustered_boxes[:, 2]).max()
            y_max = (clustered_boxes[:, 1] + clustered_boxes[:, 3]).max()
            merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return merged_boxes, fg_mask


class LogConsole(logging.Handler, QObject):
    """Консоль логирования для виджета"""
    appendLog = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__()
        QObject.__init__(self)

        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

        font = QFont("Courier New", 11)
        font.setStyleHint(QFont.Monospace)
        self.widget.setFont(font)

        self.widget.setStyleSheet("""
            QPlainTextEdit {
                border: 1px solid #444;
            }
        """)

        self.appendLog.connect(self._append_text)

    def _append_text(self, text):
        self.widget.appendPlainText(text)
        cursor = self.widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.widget.setTextCursor(cursor)

    def log(self, message):
        current_time = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"{current_time} | {message}"
        self.appendLog.emit(str(formatted_message))

    def emit(self, record):
        self.log(self.format(record))


class VideoDisplay(QLabel):
    """Виджет для отображения видео"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setText(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)

    def update_frame(self, cv_image):
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
        self.record = False
        self.video_writer = None
        self.event_recorder = None
        self.circular_buffer = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.fps = 20
        self.current_camera_index = None
        self.last_event_time = None

        self.init_ui()
        self.init_camera()

    def init_ui(self):
        self.setWindowTitle("Детектор движения с записью событий")
        self.setGeometry(100, 100, 1200, 650)

        # Общий стиль для кнопок
        button_style = """
            QPushButton {
                background-color: #e0e0e0;
                color: #000000;
                border: 1px solid #808080;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-weight: normal;
                min-width: 70px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                border-color: #404040;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
                border-color: #000000;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #a0a0a0;
                border-color: #d0d0d0;
            }
        """

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # Левая часть: видео
        video_layout = QVBoxLayout()
        video_row = QHBoxLayout()
        self.video_display = VideoDisplay("Основное видео")
        self.mask_display = VideoDisplay("Маска движения")
        video_row.addWidget(self.video_display)
        video_row.addWidget(self.mask_display)
        video_layout.addLayout(video_row)

        # Панель кнопок управления
        button_row = QHBoxLayout()
        button_row.addStretch()

        self.start_btn = QPushButton("Старт")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.setStyleSheet(button_style)
        self.start_btn.clicked.connect(self.start_capture)

        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.setStyleSheet(button_style)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_capture)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background-color: #a0a0a0; max-width: 1px; min-width: 1px; margin: 2px 4px;")

        self.record_btn = QPushButton("Запись")
        self.record_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogNoButton))
        self.record_btn.setStyleSheet(button_style)
        self.record_btn.clicked.connect(self.start_recording)

        self.stop_record_btn = QPushButton("Стоп запись")
        self.stop_record_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_record_btn.setStyleSheet(button_style)
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.clicked.connect(self.stop_recording)

        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(separator)
        button_row.addWidget(self.record_btn)
        button_row.addWidget(self.stop_record_btn)
        button_row.addStretch()

        video_layout.addLayout(button_row)

        # Правая часть: параметры
        control_layout = QVBoxLayout()

        # Группа параметров детектора
        params_group = QGroupBox("Параметры детектора")
        params_layout = QVBoxLayout()
        params_group.setMinimumWidth(280)

        # Выбор камеры
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Поиск камер...")
        self.camera_combo.setEnabled(False)
        params_layout.addWidget(self.camera_combo)

        # Слайдеры
        self.area_slider = QSlider(Qt.Horizontal)
        self.area_slider.setMinimumHeight(25)
        self.area_slider.setRange(100, 5000)
        self.area_slider.setValue(1000)
        self.area_label = QLabel(f"Порог площади: {self.area_slider.value()}")
        params_layout.addWidget(self.area_label)
        params_layout.addWidget(self.area_slider)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimumHeight(25)
        self.threshold_slider.setRange(5, 100)
        self.threshold_slider.setValue(25)
        self.threshold_label = QLabel(f"Порог дисперсии: {self.threshold_slider.value()}")
        params_layout.addWidget(self.threshold_label)
        params_layout.addWidget(self.threshold_slider)

        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setMinimumHeight(25)
        self.distance_slider.setRange(50, 500)
        self.distance_slider.setValue(200)
        self.distance_label = QLabel(f"Расстояние кластеризации: {self.distance_slider.value()}")
        params_layout.addWidget(self.distance_label)
        params_layout.addWidget(self.distance_slider)

        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimumHeight(25)
        self.kernel_slider.setRange(1, 15)
        self.kernel_slider.setValue(5)
        self.kernel_label = QLabel(f"Размер ядра: {self.kernel_slider.value()}")
        params_layout.addWidget(self.kernel_label)
        params_layout.addWidget(self.kernel_slider)

        # Выбор папки для сохранения
        path_layout = QHBoxLayout()
        default_folder = str(Path.cwd() / "recordings")
        os.makedirs(default_folder, exist_ok=True)
        self.path = QLineEdit(default_folder)
        self.path.setReadOnly(True)
        self.path_btn = QPushButton()
        self.path_btn.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        self.path_btn.clicked.connect(self.browse_folder)
        text = QLabel("Папка для логов и видео:")
        params_layout.addSpacing(5)
        params_layout.addWidget(text)
        path_layout.addWidget(self.path_btn)
        path_layout.addWidget(self.path)
        params_layout.addLayout(path_layout)

        params_group.setLayout(params_layout)

        control_layout.addWidget(params_group)
        control_layout.addStretch()

        top_layout.addLayout(video_layout, 70)
        top_layout.addLayout(control_layout, 30)

        # Консоль логов
        self.console = LogConsole(self)

        main_layout.addLayout(top_layout, 50)
        main_layout.addWidget(self.console.widget, 50)

        self.setLayout(main_layout)

        # Детектор движения
        self.detector = MotionDetector(
            min_area=self.area_slider.value(),
            sub_threshold=self.threshold_slider.value(),
            distance_threshold=self.distance_slider.value()
        )

        self.update_kernel_size()

        # Подключение сигналов слайдеров
        self.area_slider.valueChanged.connect(self.on_area_change)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        self.distance_slider.valueChanged.connect(self.on_distance_change)
        self.kernel_slider.valueChanged.connect(self.on_kernel_change)

        # Заполняем список камер
        self.populate_cameras()

        self.console.log("=== Программа запущена ===")
        self.console.log(f"Папка сохранения: {Path(default_folder).as_posix()}")

    def init_camera(self):
        """Пытается открыть камеру по умолчанию (0)"""
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.cap = cap
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 20
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.current_camera_index = 0
                self.console.log("Камера 0 открыта")
                return
        cap.release()
        self.console.log("Не удалось открыть камеру 0")

    def get_available_cameras(self, max_devices=10):
        """Сканирует доступные камеры, чтобы избежать obsensor."""
        available = []
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
        
        for i in range(max_devices):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available.append(i)
            cap.release()
        return available

    def populate_cameras(self):
        """Заполняет combobox доступными камерами."""
        cameras = self.get_available_cameras()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if cameras:
            for idx in cameras:
                self.camera_combo.addItem(f"Камера {idx}", idx)
            # Если текущий индекс ещё не выбран или недоступен, выбираем первый доступный
            if self.current_camera_index is not None and self.current_camera_index in cameras:
                index = cameras.index(self.current_camera_index)
                self.camera_combo.setCurrentIndex(index)
            else:
                self.camera_combo.setCurrentIndex(0)
                self.current_camera_index = cameras[0]
        else:
            self.camera_combo.addItem("Нет доступных камер")
        self.camera_combo.blockSignals(False)
        self.camera_combo.setEnabled(True)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)

    def on_camera_changed(self, index):
        """Обработчик смены выбранной камеры в комбобоксе."""
        if index < 0 or self.camera_combo.count() == 0:
            return
        selected_camera = self.camera_combo.currentData()
        if selected_camera is None:
            return
        if self.current_camera_index != selected_camera:
            self.switch_camera(selected_camera)

    def switch_camera(self, new_index):
        """Переключение на другую камеру с явным указанием бэкенда DirectShow."""
        # Останавливаем захват, если он активен
        was_active = False
        if self.timer.isActive():
            was_active = True
            self.timer.stop()

        # Освобождаем старую камеру
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Пытаемся открыть новую камеру
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
        
        new_cap = cv2.VideoCapture(new_index, backend)
        if not new_cap.isOpened():
            self.console.log(f"Не удалось открыть камеру {new_index}")
            # Возвращаем старую камеру, если возможно
            if self.current_camera_index is not None:
                old_cap = cv2.VideoCapture(self.current_camera_index, backend)
                if old_cap.isOpened():
                    self.cap = old_cap
                    self.console.log(f"Возврат к камере {self.current_camera_index}")
            if was_active:
                self.timer.start(int(1000 / self.fps))
            return

        # Проверяем, читается ли кадр
        ret, frame = new_cap.read()
        if not ret:
            self.console.log(f"Камера {new_index} открыта, но не даёт кадры")
            new_cap.release()
            if was_active:
                self.timer.start(int(1000 / self.fps))
            return

        # Успешно открыли
        self.cap = new_cap
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 20
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.current_camera_index = new_index
        self.console.log(f"Переключено на камеру {new_index}")

        # Если захват был активен, перезапускаем
        if was_active:
            self.timer.start(int(1000 / self.fps))

    def start_capture(self):
        """Управление захватом видео"""
        if self.cap is None or not self.cap.isOpened():
            self.console.log("Камера не открыта")
            return

        if not self.timer.isActive():
            self.timer.start(int(1000 / self.fps))
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.console.log("Захват видео запущен")
            self.init_event_system()

    def stop_capture(self):
        if self.timer.isActive():
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.console.log("Захват видео остановлен")

        if self.record:
            self.stop_recording()

    def init_event_system(self):
        """Система записи событий (клипов по движению)"""
        if self.cap and self.cap.isOpened():
            pre_frames = int(5 * self.fps)  # 5 секунд до события
            self.circular_buffer = CircularBuffer(pre_frames)
            self.event_recorder = EventRecorder(self.fps, post_seconds=5.0)

    def save_event_video(self):
        if not self.event_recorder:
            return

        try:
            frames = self.event_recorder.frames
            event_time = self.event_recorder.event_time

            if len(frames) < 10:
                return

            # Создаём подпапку events внутри выбранной папки
            events_folder = os.path.join(self.path.text(), "events")
            os.makedirs(events_folder, exist_ok=True)

            timestamp = event_time.strftime('%Y%m%d_%H%M%S')
            filename = f"clip_{timestamp}.mp4"
            filepath = os.path.join(events_folder, filename)

            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = ManagedVideoWriter(
                filepath, fourcc, self.event_recorder.fps, (width, height)
            )
            out._enforce_storage_limit()

            for frame in frames:
                out.write(frame)

            out.release()
            self.console.log(f"Сохранён клип {filename}")

        except Exception as e:
            self.console.log(f"Ошибка при сохранении события: {e}")

    def start_recording(self):
        """Ручная запись"""
        if not self.timer.isActive():
            self.console.log("Камера не активна, запись невозможна")
            return

        folder_path = self.path.text()
        os.makedirs(folder_path, exist_ok=True)

        ret, frame = self.cap.read()
        if not ret:
            self.console.log("Не удалось получить кадр для инициализации записи")
            return

        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(folder_path, filename)

        frame_size = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.video_writer = ManagedVideoWriter(
            filepath, fourcc, self.fps, frame_size
        )
        self.video_writer._enforce_storage_limit()

        if self.video_writer.isOpened():
            self.record = True
            self.record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(True)
            self.console.log(f"Начата запись: {filename}")

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.record = False
        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.console.log("Запись остановлена")

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if folder:
            self.console.log(f"Папка сохранения: {folder}")
            self.path.setText(folder)

    def update_kernel_size(self):
        value = self.kernel_slider.value()
        if value % 2 == 0:
            kernel_size = value + 1
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

    def process_frame(self):
        """Основной цикл обработки кадров"""
        ret, frame = self.cap.read()
        if not ret:
            self.console.log("Не удалось получить кадр")
            self.timer.stop()
            return

        motion_boxes, mask = self.detector.detect(frame)
        now = datetime.now()
        motion = len(motion_boxes) > 0

        # Рисуем рамки вокруг обнаруженных объектов
        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Отображаем индикатор записи
        display_frame = frame.copy()
        if self.record:
            cv2.putText(display_frame, "REC", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        self.video_display.update_frame(display_frame)
        self.mask_display.update_frame(mask_display)

        # Обработка событий (запись клипов по движению)
        if self.circular_buffer and self.event_recorder:
            self.circular_buffer.add(frame)

            if motion:
                if not self.event_recorder.is_recording:
                    pre_frames = self.circular_buffer.get_all()
                    self.event_recorder.start(now, pre_frames)
                    self.console.log("Обнаружено движение — начало записи события")

                self.event_recorder.add_motion_frame(frame, now)

            elif self.event_recorder.is_recording:
                finished = self.event_recorder.add_idle_frame(frame, now)
                if finished:
                    self.save_event_video()
                    self.event_recorder.stop()

        # Запись вручную
        if self.record and self.video_writer is not None:
            self.video_writer.write(frame)

    def closeEvent(self, event):
        """Завершение работы"""
        self.stop_capture()
        if self.cap is not None:
            self.cap.release()
        if self.console:
            self.console.deleteLater()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())