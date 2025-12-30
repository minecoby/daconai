# daconai

2024년 8월 데이콘(DACON) AI 음성탐지 대회 프로젝트

## 프로젝트 개요

오디오 파일을 분석하여 합성/변조된 음성과 실제 음성을 구분하는 이진 분류 모델

## 대회 정보

- **주최**: DACON
- **대회 기간**: 2024년 8월
- **과제**: 음성 데이터에서 가짜(fake)와 진짜(real) 음성 분류

## 주요 기술 스택

- **Python 3.x**
- **PyTorch**: 딥러닝 모델 구현
- **librosa**: 오디오 처리 및 특징 추출
- **scikit-learn**: 데이터 분할 및 평가
- **pandas/numpy**: 데이터 처리

## 모델 구조

**MLP (Multi-Layer Perceptron)**
- 입력: 15개의 MFCC 특징
- 5개의 fully connected layers (hidden dim: 128)
- Batch Normalization + Dropout (0.25)
- 출력: 2개 클래스 (fake/real)
- 활성화 함수: ReLU, Sigmoid

## 데이터 처리

1. 오디오 파일을 32kHz로 샘플링
2. MFCC (Mel-Frequency Cepstral Coefficients) 15개 추출
3. 평균값 계산하여 고정 길이 특징 벡터 생성

## 학습 설정

- Batch Size: 128
- Epochs: 6
- Learning Rate: 3e-5
- Optimizer: Adam
- Loss Function: Binary Cross Entropy
- 평가 지표: AUC (Area Under Curve)
- Train/Validation Split: 80/20

## 실행 방법

```bash
python main.py
```

## 출력

- `baseline_submit.csv`: 각 샘플에 대한 fake/real 확률 예측값

## 필요 라이브러리

```
torch
librosa
scikit-learn
numpy
pandas
tqdm
torchmetrics
```
