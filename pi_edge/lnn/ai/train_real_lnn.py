import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from ai.lnn_model import LNNAnomalyDetector
import os

print("=== 실제 데이터로 LNN 학습 시작 ===")

# 수집한 정상 데이터 불러오기
data_path = "ai/data/normal_data.pkl"
if not os.path.exists(data_path):
    print("❌ 데이터 파일을 찾을 수 없습니다.")
    exit()

with open(data_path, "rb") as f:
    sequences = pickle.load(f)

print(f"✅ 불러온 정상 데이터: {len(sequences)}개")

# 모델 생성
model = LNNAnomalyDetector(units=20)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
print("학습 시작...")

for epoch in range(30):
    total_loss = 0
    for seq in sequences:
        if len(seq) < 10:
            continue
        input_seq = torch.tensor([seq[:10]], dtype=torch.float32)  # 10 프레임 사용
        target = torch.tensor([0])  # 정상 = 0
        
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(sequences)
    print(f"Epoch {epoch+1:2d}/30, Loss: {avg_loss:.5f}")

# 모델 저장
os.makedirs("ai/models", exist_ok=True)
torch.save(model.state_dict(), "ai/models/lnn_model_real.pth")
print("\n✅ 실제 데이터로 학습 완료! 모델 저장: ai/models/lnn_model_real.pth")

# 테스트
model.eval()
test_input = torch.randn(1, 10, 5)
print("테스트 출력:", torch.softmax(model(test_input)[0], dim=0).detach().numpy())
