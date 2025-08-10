# Real-ESRGAN 훈련 옵션 설정 파일들 (Training Options)

# 📚 문서 연결성 (Document Hierarchy)

**📍 현재 위치**: L1 - 옵션 설정 가이드
**🔗 상위 문서**: [L0 - CLAUDE.md](../CLAUDE.md) - Real-ESRGAN 프로젝트 전체 가이드
**📂 관련 문서**: 
- [L2 - 모델 시스템](../realesrgan/models/models_context.md) - 훈련 설정을 활용하는 모델 시스템
- [L2 - 데이터 시스템](../realesrgan/data/data_context.md) - 설정에서 정의된 데이터 로딩 파라미터

---

이 폴더에는 Real-ESRGAN 모델 훈련 및 미세조정(Fine-tuning)을 위한 설정 파일들이 포함되어 있습니다.

## 📁 설정 파일 목록

### 🚀 훈련(Training) 설정 파일들
- **train_realesrgan_x4plus.yml**: RealESRGAN x4 모델 처음부터 훈련
- **train_realesrgan_x2plus.yml**: RealESRGAN x2 모델 처음부터 훈련  
- **train_realesrnet_x4plus.yml**: RealESRNet x4 모델 처음부터 훈련
- **train_realesrnet_x2plus.yml**: RealESRNet x2 모델 처음부터 훈련

### 🔧 미세조정(Fine-tuning) 설정 파일들
- **finetune_realesrgan_x4plus.yml**: 사전훈련된 RealESRGAN x4 모델 미세조정
- **finetune_realesrgan_x4plus_pairdata.yml**: 페어 데이터를 이용한 RealESRGAN x4 모델 미세조정

## ⚙️ 주요 설정 옵션 설명

### 일반 설정 (General Settings)
- **name**: 실험 이름
- **model_type**: 모델 타입 (RealESRGANModel, RealESRNetModel)
- **scale**: 업스케일링 배율 (2 또는 4)
- **num_gpu**: 사용할 GPU 수 (auto로 설정 시 자동 감지)
- **manual_seed**: 재현 가능한 결과를 위한 시드값

### 데이터 증강 설정 (Data Degradation)
Real-ESRGAN은 실제 저화질 이미지를 시뮬레이션하기 위해 두 단계의 열화(degradation) 과정을 사용합니다:

#### 1차 열화 과정
- **resize_prob**: 크기 조정 확률 [업스케일, 다운스케일, 유지]
- **resize_range**: 크기 조정 범위
- **gaussian_noise_prob**: 가우시안 노이즈 추가 확률
- **noise_range**: 노이즈 강도 범위
- **jpeg_range**: JPEG 압축 품질 범위 [30-95]

#### 2차 열화 과정  
- **second_blur_prob**: 추가 블러 적용 확률
- **resize_prob2**: 2차 크기 조정 확률
- **gaussian_noise_prob2**: 2차 노이즈 추가 확률

### USM 샤프닝 설정
- **l1_gt_usm**: L1 손실용 GT 이미지에 USM 적용
- **percep_gt_usm**: Perceptual 손실용 GT 이미지에 USM 적용  
- **gan_gt_usm**: GAN 손실용 GT 이미지에 USM 적용

## 📋 모델별 특징

### RealESRGAN vs RealESRNet
- **RealESRGAN**: GAN 기반으로 더 선명한 결과, 아티팩트 가능성 있음
- **RealESRNet**: MSE 기반으로 안정적이지만 상대적으로 부드러운 결과

### 스케일별 차이점
- **x2 모델**: 2배 업스케일링, 빠른 처리 속도
- **x4 모델**: 4배 업스케일링, 더 극적인 해상도 향상

## 🛠️ 사용 방법

### 새로운 모델 훈련
```bash
python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml
```

### 사전훈련된 모델 미세조정
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml
```

## ⚠️ 주의사항

1. **GPU 메모리**: 배치 크기와 GPU 수를 시스템 사양에 맞게 조정하세요
2. **데이터셋**: 고품질 데이터셋이 필수입니다 (DF2K, OST 등)
3. **훈련 시간**: x4 모델은 완전 훈련 시 며칠이 소요될 수 있습니다
4. **미세조정**: 특정 도메인(애니메이션, 얼굴 등)에 맞게 미세조정하면 더 좋은 결과를 얻을 수 있습니다

## 📚 더 자세한 정보

훈련에 대한 자세한 내용은 `docs/Training.md` 파일을 참고하세요.