import sys, os, cv2, logging, numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QTextCursor, QFont
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, QFileDialog, QStyle,
                             QPushButton, QSlider, QHBoxLayout, QGroupBox, QLineEdit, QPlainTextEdit,
                             QComboBox, QSpinBox, QCheckBox)
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import threading


class CircularBuffer:
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
    def __init__(self, pre_frames, post_frames, fps, frame_size):
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.fps = fps
        self.frame_size = frame_size
        self.post_buffer = []
        self.is_recording = False
        self.frames_to_record = 0
        self.event_time = None

    def start_recording(self, event_time):
        self.is_recording = True
        self.frames_to_record = self.post_frames
        self.post_buffer = []
        self.event_time = event_time

    def add_frame(self, frame):
        if self.is_recording and self.frames_to_record > 0:
            self.post_buffer.append(frame.copy())
            self.frames_to_record -= 1
            if self.frames_to_record == 0:
                self.is_recording = False
                return True
        return False

    def get_complete_event(self, pre_frames):
        return pre_frames + self.post_buffer


class StorageManager:
    def __init__(self, storage_path, max_size_mb=1000):
        self.storage_path = Path(storage_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.events_path = self.storage_path / "events"
        self.events_path.mkdir(parents=True, exist_ok=True)

    def check_and_clean(self):
        try:
            total_size = 0
            files = []

            for file_path in self.events_path.glob("*.mp4"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_time = file_path.stat().st_mtime
                    files.append((file_time, file_path, file_size))
                    total_size += file_size

            if total_size > self.max_size_bytes:
                files.sort()

                for file_time, file_path, file_size in files:
                    if total_size <= self.max_size_bytes:
                        break

                    try:
                        file_path.unlink()
                        total_size -= file_size
                    except:
                        pass

            return len(files), total_size / (1024 * 1024)

        except Exception as e:
            return 0, 0

    def get_storage_info(self):
        return self.check_and_clean()


class MotionDetector:
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
        self.storage_manager = None
        self.last_event_time = None
        self.init_ui()
        self.init_camera()

    def init_ui(self):
        self.setWindowTitle("")
        self.setGeometry(100, 100, 1200, 650)

        self.setStyleSheet("""
                    QPushButton {
                    background-color: #e0e0e0;
                    color: #000000;
                    border: 1px solid #808080;
                    border-radius: 4px;
                    font-size: 24px;
                    padding: 10px;
                        min-width: 40px;
                        min-height: 20px;
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
                QSpinBox {
                    padding: 3px;
                    border: 1px solid #808080;
                    border-radius: 3px;
                    min-height: 20px;
                }
                QCheckBox {
                    spacing: 5px;
                }
                """)

        main_layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        video_layout = QVBoxLayout()
        video_row = QHBoxLayout()
        self.video_display = VideoDisplay("Основное видео")
        self.mask_display = VideoDisplay("Маска движения")
        video_row.addWidget(self.video_display)
        video_row.addWidget(self.mask_display)
        video_layout.addLayout(video_row)

        control_layout = QVBoxLayout()

        button_group = QGroupBox()
        button_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 8px;
            }
        """)

        button_layout = QHBoxLayout(button_group)
        self.start_btn = QPushButton()
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.stop_btn.setEnabled(False)

        self.record_btn = QPushButton()
        self.record_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogNoButton))
        self.stop_record_btn = QPushButton()
        self.stop_record_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.stop_record_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #8b3a3a;
                        }
                        QPushButton:hover {
                            background-color: #a54545;
                        }
                        QPushButton:disabled {
                    background-color: #f0f0f0;
                    color: #a0a0a0;
                    border-color: #d0d0d0;
                }
                    """)
        self.stop_record_btn.setEnabled(False)

        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.stop_record_btn)
        button_layout.addStretch()

        params_group = QGroupBox("Параметры детектора")
        params_layout = QVBoxLayout()
        params_group.setMinimumWidth(280)

        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Поиск камер...")
        self.camera_combo.setEnabled(False)
        self.populate_cameras()
        params_layout.addWidget(self.camera_combo)
        self.current_camera_index = 0

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

        event_group = QGroupBox("Сохранение событий")
        event_layout = QVBoxLayout()

        pre_layout = QHBoxLayout()
        pre_layout.addWidget(QLabel("Секунд ДО:"))
        self.pre_spin = QSpinBox()
        self.pre_spin.setRange(1, 30)
        self.pre_spin.setValue(5)
        pre_layout.addWidget(self.pre_spin)
        pre_layout.addStretch()
        event_layout.addLayout(pre_layout)

        post_layout = QHBoxLayout()
        post_layout.addWidget(QLabel("Секунд ПОСЛЕ:"))
        self.post_spin = QSpinBox()
        self.post_spin.setRange(1, 30)
        self.post_spin.setValue(5)
        post_layout.addWidget(self.post_spin)
        post_layout.addStretch()
        event_layout.addLayout(post_layout)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Мин. интервал (сек):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(5)
        interval_layout.addWidget(self.interval_spin)
        interval_layout.addStretch()
        event_layout.addLayout(interval_layout)

        storage_layout = QHBoxLayout()
        storage_layout.addWidget(QLabel("Лимит (MB):"))
        self.storage_spin = QSpinBox()
        self.storage_spin.setRange(100, 10000)
        self.storage_spin.setValue(1000)
        self.storage_spin.setSingleStep(100)
        storage_layout.addWidget(self.storage_spin)
        storage_layout.addStretch()
        event_layout.addLayout(storage_layout)

        self.enable_events = QCheckBox("Сохранять видео событий")
        self.enable_events.setChecked(True)
        event_layout.addWidget(self.enable_events)

        event_group.setLayout(event_layout)

        path_layout = QHBoxLayout()
        default_folder = str(Path.cwd() / "recordings")
        os.makedirs(default_folder, exist_ok=True)
        self.path = QLineEdit(default_folder)
        self.path.setReadOnly(True)
        self.path_btn = QPushButton()
        self.path_btn.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        text = QLabel("Папка для логов и видео:")
        params_layout.addSpacing(5)
        params_layout.addWidget(text)
        path_layout.addWidget(self.path_btn)
        path_layout.addWidget(self.path)
        params_layout.addLayout(path_layout)

        params_group.setLayout(params_layout)

        control_layout.addWidget(button_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(event_group)
        control_layout.addStretch()

        top_layout.addLayout(video_layout, 70)
        top_layout.addLayout(control_layout, 30)

        self.console = LogConsole(self)

        main_layout.addLayout(top_layout, 50)
        main_layout.addWidget(self.console.widget, 50)

        self.setLayout(main_layout)

        self.detector = MotionDetector(
            min_area=self.area_slider.value(),
            sub_threshold=self.threshold_slider.value(),
            distance_threshold=self.distance_slider.value()
        )

        self.update_kernel_size()

        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        self.record_btn.clicked.connect(self.start_recording)
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.path_btn.clicked.connect(self.browse_folder)

        self.area_slider.valueChanged.connect(self.on_area_change)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        self.distance_slider.valueChanged.connect(self.on_distance_change)
        self.kernel_slider.valueChanged.connect(self.on_kernel_change)

        self.pre_spin.valueChanged.connect(self.init_event_system)
        self.post_spin.valueChanged.connect(self.init_event_system)
        self.storage_spin.valueChanged.connect(self.update_storage)
        self.path.textChanged.connect(self.init_event_system)

        self.console.log("=== Программа запущена ===")
        self.console.log(f"Папка сохранения: {Path(default_folder).as_posix()}")

    def init_event_system(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 20

            pre_frames = int(self.pre_spin.value() * fps)
            post_frames = int(self.post_spin.value() * fps)
            frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            self.circular_buffer = CircularBuffer(pre_frames)
            self.event_recorder = EventRecorder(pre_frames, post_frames, fps, frame_size)
            self.storage_manager = StorageManager(self.path.text(), self.storage_spin.value())

    def update_storage(self):
        if self.storage_manager:
            self.storage_manager.max_size_bytes = self.storage_spin.value() * 1024 * 1024

    def save_event_video(self, pre_frames, event_time):
        if not self.event_recorder or not self.storage_manager:
            return

        try:
            all_frames = self.event_recorder.get_complete_event(pre_frames)

            if len(all_frames) < 10:
                return

            timestamp = event_time.strftime('%Y%m%d_%H%M%S')
            filename = f"event_{timestamp}_pre{self.pre_spin.value()}s_post{self.post_spin.value()}s.mp4"
            filepath = self.storage_manager.events_path / filename

            height, width = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(filepath), fourcc, self.event_recorder.fps, (width, height))

            for frame in all_frames:
                out.write(frame)

            out.release()

            self.console.log(f"Событие: {filename}")

        except Exception as e:
            self.console.log(f"Ошибка при сохранении события: {e}")

    def get_available_cameras(self, max_devices=10):
        available = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available.append(i)
            cap.release()
        return available

    def populate_cameras(self):
        cameras = self.get_available_cameras()
        self.camera_combo.clear()
        if cameras:
            for idx in cameras:
                self.camera_combo.addItem(f"Камера {idx}", idx)
            current_index = getattr(self, 'current_camera_index', 0)
            if current_index in cameras:
                self.camera_combo.setCurrentIndex(cameras.index(current_index))
            else:
                self.camera_combo.setCurrentIndex(0)
                current_index = cameras[0]
        else:
            self.camera_combo.addItem("Нет доступных камер")
        self.camera_combo.setEnabled(True)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)

    def on_camera_changed(self, index):
        if index < 0 or self.camera_combo.count() == 0:
            return
        selected_camera = self.camera_combo.currentData()
        if selected_camera is None:
            return
        if self.current_camera_index != index:
            self.switch_camera(selected_camera)

    def switch_camera(self, new_index):
        """Переключает на камеру с указанным индексом."""
        if self.current_camera_index == new_index:
            return  # уже на этой камере

        # Останавливаем захват, если он активен
        was_active = False
        if hasattr(self, 'timer') and self.timer.isActive():
            was_active = True
            self.timer.stop()

        # Освобождаем старую камеру
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Открываем новую камеру
        new_cap = cv2.VideoCapture(new_index)
        if not new_cap.isOpened():
            self.console.log(f"Не удалось открыть камеру {new_index}")
            # Пытаемся вернуться к предыдущей камере (если была)
            if self.current_camera_index is not None:
                self.console.log("Оставляем текущую камеру")
            return

        # Устанавливаем параметры (можно скорректировать под ваши нужды)
        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        new_cap.set(cv2.CAP_PROP_FPS, 20)

        self.cap = new_cap
        self.current_camera_index = new_index
        self.console.log(f"Переключено на камеру {new_index}")

        # Если захват был активен, перезапускаем его
        if was_active:
            self.timer.start(50)   # используйте тот же интервал, что и в start_capture
            self.init_event_system()   # переинициализируем буферы под новую камеру

    def start_recording(self):
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

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть камеру.")
            self.video_display.setText("Не удалось открыть камеру")
            self.mask_display.setText("Не удалось открыть камеру")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def start_capture(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.console.log("Камера не открыта")
            return

        if not self.timer.isActive():
            self.timer.start(50)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.console.log("Захват видео запущен")

            self.init_event_system()
            self.last_event_time = None

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.console.log("Не удалось получить кадр")
            self.timer.stop()
            return

        motion_boxes, mask = self.detector.detect(frame)

        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        display_frame = frame.copy()
        if self.record:
            cv2.putText(display_frame, "REC", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        self.video_display.update_frame(display_frame)
        self.mask_display.update_frame(mask_display)

        if self.enable_events.isChecked() and self.circular_buffer and self.event_recorder:
            self.circular_buffer.add(frame)

            if self.event_recorder.is_recording:
                finished = self.event_recorder.add_frame(frame)
                if finished:
                    pre_frames = self.circular_buffer.get_all()
                    self.save_event_video(pre_frames, self.event_recorder.event_time)

            current_time = datetime.now()
            if len(motion_boxes) > 0:
                should_record = False
                if self.last_event_time is None:
                    should_record = True
                else:
                    time_diff = (current_time - self.last_event_time).total_seconds()
                    if time_diff >= self.interval_spin.value():
                        should_record = True

                if should_record and not self.event_recorder.is_recording:
                    self.event_recorder.start_recording(current_time)
                    self.last_event_time = current_time

        if self.record and self.video_writer is not None:
            self.video_writer.write(frame)

    def stop_capture(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.console.log("Захват видео остановлен")

        if self.record:
            self.stop_recording()

    def closeEvent(self, event):
        self.stop_capture()
        if hasattr(self, 'cap'):
            self.cap.release()

        if hasattr(self, 'console'):
            self.console.deleteLater()
            self.console = None

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())