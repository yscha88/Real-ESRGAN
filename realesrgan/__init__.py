#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-ESRGAN 패키지 초기화 모듈 (Real-ESRGAN Package Initialization Module)
===============================================================================

Real-ESRGAN (Real-Enhanced Super-Resolution Generative Adversarial Networks)은
실용적인 이미지 복원 및 초해상도(Super-Resolution) 향상을 위한 딥러닝 패키지입니다.

주요 기능 (Main Features):
--------------------------
1. 실제 이미지에 특화된 초해상도 향상 알고리즘
2. 다양한 네트워크 아키텍처 지원 (ESRGAN, Real-ESRNet 등)
3. 블라인드 초해상도(Blind Super-Resolution) 지원
4. 실제 저화질 이미지 복원에 최적화된 손실 함수
5. 타일 기반 처리로 대용량 이미지 처리 가능

패키지 구조 (Package Structure):
---------------------------------
- archs/: 신경망 아키텍처 모듈 (Network Architecture Modules)
  * discriminator_arch.py: 판별기 네트워크 구조
  * srvgg_arch.py: SRVgg 생성기 네트워크 구조
  
- data/: 데이터셋 및 데이터 로더 모듈 (Dataset and DataLoader Modules)
  * realesrgan_dataset.py: Real-ESRGAN용 데이터셋 클래스
  * realesrgan_paired_dataset.py: 페어 데이터셋 클래스
  
- models/: 훈련 및 추론 모델 모듈 (Training and Inference Model Modules)
  * realesrgan_model.py: Real-ESRGAN 모델 클래스
  * realesrnet_model.py: Real-ESRNet 모델 클래스
  
- utils.py: 유틸리티 함수 및 RealESRGANer 클래스
- train.py: 모델 훈련 스크립트
- version.py: 버전 정보

사용 방법 (Usage):
------------------
```python
from realesrgan import RealESRGANer
from realesrgan.archs.rrdbnet_arch import RRDBNet

# 모델 초기화
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

# 이미지 업샘플링
import cv2
img = cv2.imread('input.jpg', cv2.IMREAD_UNCHANGED)
output, _ = upsampler.enhance(img, outscale=4)
cv2.imwrite('output.jpg', output)
```

개발자 정보 (Developer Information):
------------------------------------
- 원본 저장소: https://github.com/xinntao/Real-ESRGAN
- 논문: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
- 라이선스: BSD 3-Clause License

버전 정보 (Version Information):
--------------------------------
현재 버전에 대한 자세한 정보는 version.py 파일을 참조하세요.
"""

# flake8: noqa
from .archs import *
from .data import *
from .models import *
from .utils import *
from .version import *
