import cv2
import numpy as np

class AdvancedMotionDetector:
    '''
    Docstring для AdvancedMotionDetector
    '''
    def __init__(self, min_area=1000, threshold=25):
        '''
        Docstring для __init__
        
        :param min_area: параметр отвечающий за минимальный размер bbox маски
        :param threshold: порог дисперсии, пиксели с дисперсией ниже этого порога считаются фоновыми. 
        '''
        # TODO: сделать ползунок для регулировки в приложении и автоматическая калибровка при использовании флажка
        self.min_area = min_area
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=threshold, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # ядро для применения поиска
        
    def detect(self, frame):
        '''
        Docstring для detect
        
        :param frame: отдельный кадр
        '''
        # Фоновая субтракция
        fg_mask = self.sub.apply(frame)
        
        # Удаление шума
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        # Заполнение отверстий
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        # Усреднение значений пикселей
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Поиск контура
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_boxes = []
        for contour in contours:
            # Площадь многоугольника ограниченного контуром
            area = cv2.contourArea(contour)
            # Отсеивание по размеру
            if area > self.min_area:
                # Поиск крайних точек TODO: улучшить алгоритм поиска и объединения bbox по порогу пересечения
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))
                
        return motion_boxes, fg_mask

# Определение; TODO: добавить настройку логирования
detector = AdvancedMotionDetector()
# TODO: настроить выбор устройства
cap = cv2.VideoCapture(0)

# Основной цикл
while True:
    # Чтение потока
    ret, frame = cap.read()
    if not ret:
        break
        
    # Получение bbox
    motion_boxes, mask = detector.detect(frame)
    
    # Рендеринг
    for (x, y, w, h) in motion_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Motion", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Motion Detection', frame)
    cv2.imshow('Motion Mask', mask)
    # Выход, остановка
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # TODO: Добавить логирование в цикл

# После выхода
cap.release()
cv2.destroyAllWindows()