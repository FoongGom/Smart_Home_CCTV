# 🏠 Smart Home CCTV + AI System

> Raspberry Pi 엣지 컴퓨팅 + PC 서버 하이브리드 구조의 스마트 홈 보안 시스템

## 📌 프로젝트 개요
- Raspberry Pi에서 **카메라 영상 수집 + LNN(액체 신경망)** 추론 수행
- PC에서 **Flask 웹 서버 + MariaDB** 관리
- 실시간 이벤트 감지 및 보안 모니터링 시스템

## 🏗 아키텍처
[Raspberry Pi - Edge]               [PC - Server]
│                                   │
Camera ──▶ LNN 추론 ─────────────▶ Flask API
│                                   │
└────────── 이벤트 데이터 ─────────▶ MariaDB
text- **pi_edge/**: Raspberry Pi 전용 코드 (Camera + LNN)
- **server/**: PC에서 실행되는 Flask + MariaDB 서버
- **shared/**: 공통 모듈

## 🛠 기술 스택
- **Edge (Pi)**: Python, OpenCV, LNN
- **Server (PC)**: Flask, MariaDB, SQLAlchemy
- **기타**: systemd, Docker (예정)

## 📁 프로젝트 구조
smart_home_security/
├── pi_edge/          # Raspberry Pi 전용
│   ├── camera/
│   └── lnn/
├── server/           # PC 서버 (Flask + MariaDB)
│   ├── app/
│   ├── config.py
│   └── create_db.sql
├── shared/
├── assets/
└── data/
text## 🚀 실행 방법

### Raspberry Pi (Edge)
```bash
cd pi_edge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
PC 서버
Bashcd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
📝 현재 상태 및 계획

 코드 구조를 Pi Edge / PC Server로 분리
 MariaDB로 DB 이전 준비
 LNN 모델 경량화 및 실시간 추론 최적화
 Flask 대시보드 구현
 실시간 알림 기능 (Telegram 등)

📫 연락처

GitHub: FoongGom
