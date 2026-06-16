import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ai.lnn_model import LNNAnomalyDetector
from ai.feature_extractor import FeatureExtractor
import time
import os
from datetime import datetime

print("=== LNN 모델 학습 준비 ===")

# 모델 생성
model = LNNAnomalyDetector(units=20)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

extractor = FeatureExtractor()
sequence_data = []
labels = []

print("✅ 정상 데이터 수집 시작 (dummy 데이터로 학습)...")
print("Ctrl+C로 수집 종료")

try:
    for i in range(500):
        # 정상 패턴 dummy 데이터
        dummy_seq = np.random.randn(10, 5) * 0.6 + 0.5   # 정상적인 움직임 패턴
        sequence_data.append(dummy_seq)
        labels.append(0)  # 0 = 정상
        
        if i % 100 == 0:
            print(f"수집된 정상 시퀀스: {i+1}/500개")
        time.sleep(0.03)
        
except KeyboardInterrupt:
    print(f"\n총 {len(sequence_data)}개 데이터 수집 완료")

# 학습
print("\n=== LNN 모델 학습 시작 ===")
model.train()

for epoch in range(20):
    total_loss = 0
    for seq in sequence_data:
        input_seq = torch.tensor([seq], dtype=torch.float32)
        target = torch.tensor([0])   # 정상
        
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(sequence_data)
    print(f"Epoch {epoch+1:2d}/20, Loss: {avg_loss:.4f}")

# 모델 저장
os.makedirs("ai/models", exist_ok=True)
torch.save(model.state_dict(), "ai/models/lnn_model.pth")
print("\n✅ 모델 학습 완료 및 저장: ai/models/lnn_model.pth")

# 테스트
model.eval()
test_input = torch.randn(1, 10, 5)
print("테스트 출력:", torch.softmax(model(test_input)[0], dim=0).detach().numpy())
