import torch
import torch.nn as nn
from ncps.torch import LTC
import numpy as np
import os

class LNNAnomalyDetector(nn.Module):
    def __init__(self, input_size=5, units=24):   # units = hidden neurons
        super().__init__()
        self.lnn = LTC(input_size=input_size, units=units, return_sequences=True)
        self.fc = nn.Linear(units, 2)  # 0: 정상, 1: 이상
        
    def forward(self, x):
        # x: (batch, sequence_length, features)
        lnn_out, _ = self.lnn(x)
        out = self.fc(lnn_out[:, -1, :])  # 마지막 타임스텝
        return out

def load_model():
    model = LNNAnomalyDetector(input_size=5, units=24)
    model.eval()
    print("✅ LNN 모델 생성 완료 (units=24)")
    return model

if __name__ == "__main__":
    model = load_model()
    # 간단 테스트
    test_input = torch.randn(1, 10, 5)  # batch, seq, features
    output = model(test_input)
    print("✅ 테스트 출력:", output.detach().numpy())
