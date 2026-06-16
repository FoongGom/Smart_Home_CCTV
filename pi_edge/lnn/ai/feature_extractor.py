import cv2
import numpy as np
from collections import deque

class FeatureExtractor:
    def __init__(self, maxlen=30):
        self.motion_history = deque(maxlen=maxlen)
        
    def extract_features(self, frame, motion_detected, motion_score):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        features = {
            'motion_score': float(motion_score),
            'motion_detected': motion_detected,
            'mean_brightness': float(np.mean(gray)),
            'std_brightness': float(np.std(gray)),
        }
        
        # Optical Flow (움직임 방향/속도)
        if len(self.motion_history) > 0:
            prev_gray = self.motion_history[-1].get('gray')
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                features['optical_flow_mean'] = float(np.mean(magnitude))
                features['optical_flow_max'] = float(np.max(magnitude))
            else:
                features['optical_flow_mean'] = 0.0
                features['optical_flow_max'] = 0.0
        else:
            features['optical_flow_mean'] = 0.0
            features['optical_flow_max'] = 0.0
        
        features['gray'] = gray
        self.motion_history.append(features)
        
        # LNN에 입력할 feature vector (5차원)
        feature_vector = np.array([
            features['motion_score'],
            features['mean_brightness'],
            features['std_brightness'],
            features['optical_flow_mean'],
            features['optical_flow_max']
        ])
        
        return features, feature_vector

if __name__ == "__main__":
    print("✅ Feature Extractor 준비 완료")
