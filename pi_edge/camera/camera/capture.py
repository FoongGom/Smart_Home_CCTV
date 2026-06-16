import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import numpy as np
import torch
import requests
from datetime import datetime
from picamera2 import Picamera2
from ai.feature_extractor import FeatureExtractor
from ai.lnn_model import LNNAnomalyDetector

# === PC 서버 주소 ===
SERVER_URL = "http://172.16.113.66:5000/api/event"

class MotionDetector:
    def __init__(self, min_area=1600, threshold=32, cooldown=0.45):
        self.min_area = min_area
        self.threshold = threshold
        self.cooldown = cooldown
        self.prev_frame = None
        self.last_detection_time = 0
        
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0, frame, gray
        
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        motion_score = 0
        current_time = time.time()
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if current_time - self.last_detection_time < self.cooldown:
                continue
            motion_detected = True
            motion_score += area
            self.last_detection_time = current_time
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.prev_frame = gray
        return motion_detected, motion_score, frame, gray


if __name__ == "__main__":
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (480, 360), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    detector = MotionDetector()
    extractor = FeatureExtractor()
    lnn_model = LNNAnomalyDetector(units=20)
    
    model_path = "ai/models/lnn_model_real.pth"
    if os.path.exists(model_path):
        lnn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✅ LNN 모델 로드 완료")
    
    lnn_model.eval()
    
    print("🚀 Raspberry Pi - Edge Inference 시작")
    print(f"이벤트 전송 주소: {SERVER_URL}")
    
    frame_skip = 0
    sequence = []
    
    try:
        while True:
            frame = picam2.capture_array()
            frame_skip += 1
            
            motion, score, processed, _ = detector.process_frame(frame)
            _, feature_vector = extractor.extract_features(processed, motion, score)
            
            sequence.append(feature_vector)
            if len(sequence) > 10:
                sequence.pop(0)
            
            anomaly_detected = False
            anomaly_score = 0.0
            
            if score > 18000 or (frame_skip % 4 == 0 and len(sequence) >= 10 and score > 5000):
                input_seq = torch.tensor(np.array([sequence]), dtype=torch.float32)
                with torch.no_grad():
                    output = lnn_model(input_seq)
                    anomaly_score = torch.softmax(output[0], dim=0)[1].item()
                if anomaly_score > 0.55 or score > 18000:
                    anomaly_detected = True
            
            if anomaly_detected:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"event_{timestamp}.jpg"
                cv2.imwrite(image_path, processed)
                
                try:
                    requests.post(SERVER_URL, json={
                        "event_type": "anomaly",
                        "confidence": float(anomaly_score),
                        "motion_score": float(score),
                        "image_path": image_path,
                        "description": "Pi에서 LNN 감지"
                    }, timeout=1)
                    print(f"🚨 PC로 이벤트 전송 완료! Score: {score:.0f}")
                except Exception as e:
                    print(f"전송 실패: {e}")
            
            cv2.imshow("LNN Edge", processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.07)
            
    except KeyboardInterrupt:
        print("\nPi 프로그램 종료")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
