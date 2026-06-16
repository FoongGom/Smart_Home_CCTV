import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pickle
from picamera2 import Picamera2
from ai.feature_extractor import FeatureExtractor
import time

print("=== 정상 데이터 수집 모드 ===")
print("카메라 앞에서 **일반적인 움직임** (손 움직이기, 몸 움직이기 등)을 1~2분 정도 해주세요.")
print("Ctrl+C로 수집 종료")

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (400, 300), "format": "RGB888"})
picam2.configure(config)
picam2.start()

extractor = FeatureExtractor()
sequences = []
frame_skip = 0

try:
    while True:
        frame = picam2.capture_array()
        frame_skip += 1
        
        # 간단 모션 처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        _, feature_vector = extractor.extract_features(frame, True, 1000)  # dummy score
        
        if frame_skip % 3 == 0:   # 3프레임마다 저장
            sequences.append(feature_vector.tolist())
        
        cv2.imshow("Normal Data Collection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        time.sleep(0.05)
        
except KeyboardInterrupt:
    print(f"\n총 {len(sequences)}개 feature 수집 완료")

finally:
    picam2.stop()
    cv2.destroyAllWindows()

# 저장
os.makedirs("ai/data", exist_ok=True)
with open("ai/data/normal_data.pkl", "wb") as f:
    pickle.dump(sequences, f)

print(f"✅ 정상 데이터 저장 완료: ai/data/normal_data.pkl ({len(sequences)}개)")
