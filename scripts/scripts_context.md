# Real-ESRGAN Scripts 분석 문서

# 📚 문서 연결성 (Document Hierarchy)

**📍 현재 위치**: L1 - 스크립트 시스템 가이드
**🔗 상위 문서**: [L0 - CLAUDE.md](../CLAUDE.md) - Real-ESRGAN 프로젝트 전체 가이드
**📂 관련 문서**: 
- [L2 - 데이터 시스템](../realesrgan/data/data_context.md) - 전처리된 데이터 로딩 시스템
- [L2 - 모델 시스템](../realesrgan/models/models_context.md) - 전처리 데이터를 활용하는 훈련 시스템

---

## 스크립트 개요

`scripts` 폴더는 Real-ESRGAN 모델 학습 및 추론을 위한 데이터 전처리(Data Preprocessing) 스크립트들을 포함하고 있습니다. 이 스크립트들은 주로 이미지 데이터셋 준비, 메타데이터 생성, 모델 변환 등의 작업을 수행하며, 머신러닝 파이프라인에서 중요한 전처리 단계를 담당합니다.

주요 역할:
- 대용량 이미지를 작은 패치로 분할하여 메모리 효율적인 학습 환경 구성
- 다양한 스케일의 이미지 생성으로 멀티스케일 학습 데이터 준비
- 학습용 메타정보 파일 생성으로 데이터 로딩 최적화
- PyTorch 모델을 ONNX 형식으로 변환하여 배포 환경 지원

## 각 스크립트별 상세 분석

### 1. extract_subimages.py - 서브이미지 추출기

**목적과 기능:**
- 대용량 고해상도 이미지를 작은 크기의 서브이미지(sub-images)로 분할
- 슬라이딩 윈도우(sliding window) 방식으로 겹치는 패치 생성
- 멀티스레딩을 통한 고속 처리

**주요 파라미터:**
- `--input`: 입력 폴더 경로 (기본값: datasets/DF2K/DF2K_HR)
- `--output`: 출력 폴더 경로 (기본값: datasets/DF2K/DF2K_HR_sub)
- `--crop_size`: 크롭할 패치 크기 (기본값: 480)
- `--step`: 슬라이딩 윈도우 이동 간격 (기본값: 240)
- `--thresh_size`: 최소 패치 크기 임계값 (기본값: 0)
- `--n_thread`: 멀티스레딩 스레드 수 (기본값: 20)
- `--compression_level`: PNG 압축 레벨 0-9 (기본값: 3)

**입력/출력 형식:**
- 입력: 고해상도 이미지 파일들 (PNG, JPG 등)
- 출력: 패치 단위로 분할된 이미지 파일들 (`원본파일명_s001.png` 형식)

**사용 예시:**
```bash
python scripts/extract_subimages.py \
    --input datasets/DF2K/DF2K_HR \
    --output datasets/DF2K/DF2K_HR_sub \
    --crop_size 480 \
    --step 240 \
    --n_thread 8
```

### 2. generate_meta_info.py - 메타정보 생성기 (GT 전용)

**목적과 기능:**
- Ground Truth(GT) 이미지들에 대한 메타정보 텍스트 파일 생성
- 여러 폴더의 이미지를 하나의 메타 파일로 통합
- 이미지 유효성 검증 기능 포함

**주요 파라미터:**
- `--input`: 입력 폴더 목록 (기본값: DF2K/DF2K_HR, DF2K/DF2K_multiscale)
- `--root`: 루트 폴더 목록 (입력 폴더와 동일한 개수)
- `--meta_info`: 메타정보 텍스트 파일 경로
- `--check`: 이미지 유효성 검사 활성화 플래그

**입력/출력 형식:**
- 입력: 이미지 파일들이 포함된 폴더
- 출력: 상대 경로 목록이 포함된 텍스트 파일

**사용 예시:**
```bash
python scripts/generate_meta_info.py \
    --input datasets/DF2K/DF2K_HR datasets/DF2K/DF2K_multiscale \
    --root datasets/DF2K datasets/DF2K \
    --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt \
    --check
```

### 3. generate_meta_info_pairdata.py - 페어 데이터 메타정보 생성기

**목적과 기능:**
- GT(Ground Truth)와 LQ(Low Quality) 이미지 쌍에 대한 메타정보 생성
- Super-resolution 학습용 페어 데이터셋 관리

**주요 파라미터:**
- `--input`: GT 폴더와 LQ 폴더 경로 (2개 필수)
- `--root`: 각 폴더의 루트 경로 (2개, None 가능)
- `--meta_info`: 메타정보 텍스트 파일 경로

**입력/출력 형식:**
- 입력: GT 이미지 폴더와 LQ 이미지 폴더 (동일한 개수의 파일 필요)
- 출력: GT와 LQ 이미지 쌍의 상대 경로가 포함된 텍스트 파일

**사용 예시:**
```bash
python scripts/generate_meta_info_pairdata.py \
    --input datasets/DF2K/DIV2K_train_HR_sub datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub \
    --meta_info datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt
```

### 4. generate_multiscale_DF2K.py - 멀티스케일 이미지 생성기

**목적과 기능:**
- 원본 고해상도 이미지에서 다양한 스케일의 이미지 생성
- LANCZOS 리샘플링을 사용한 고품질 다운스케일링
- DF2K 데이터셋(DIV2K + Flickr2K)용 최적화

**주요 파라미터:**
- `--input`: 입력 고해상도 이미지 폴더 (기본값: datasets/DF2K/DF2K_HR)
- `--output`: 출력 멀티스케일 이미지 폴더 (기본값: datasets/DF2K/DF2K_multiscale)

**처리 스케일:**
- 0.75배 스케일 (`파일명T0.png`)
- 0.5배 스케일 (`파일명T1.png`)
- 1/3배 스케일 (`파일명T2.png`)
- 최단변 400px 고정 스케일 (`파일명T3.png`)

**입력/출력 형식:**
- 입력: 고해상도 이미지 파일들
- 출력: 4가지 스케일의 다운스케일된 PNG 이미지

**사용 예시:**
```bash
python scripts/generate_multiscale_DF2K.py \
    --input datasets/DF2K/DF2K_HR \
    --output datasets/DF2K/DF2K_multiscale
```

### 5. pytorch2onnx.py - 모델 변환기

**목적과 기능:**
- PyTorch 모델을 ONNX(Open Neural Network Exchange) 형식으로 변환
- RRDBNet 아키텍처에 특화된 변환 도구
- 모델 배포 및 추론 최적화를 위한 형식 변환

**주요 파라미터:**
- `--input`: 입력 PyTorch 모델 경로 (.pth 파일)
- `--output`: 출력 ONNX 모델 경로 (.onnx 파일)
- `--params`: params_ema 대신 params 사용 플래그

**모델 구성:**
- 입력 채널: 3 (RGB)
- 출력 채널: 3 (RGB)
- 특성 채널: 64
- RRDB 블록 수: 23
- 성장 채널: 32
- 업스케일 팩터: 4배

**입력/출력 형식:**
- 입력: PyTorch 모델 파일 (.pth)
- 출력: ONNX 모델 파일 (.onnx)

**사용 예시:**
```bash
python scripts/pytorch2onnx.py \
    --input experiments/pretrained_models/RealESRGAN_x4plus.pth \
    --output realesrgan-x4.onnx
```

## 데이터 전처리 워크플로우

Real-ESRGAN 모델 학습을 위한 권장 데이터 전처리 순서:

### 1단계: 멀티스케일 데이터 생성
```bash
# 고해상도 이미지에서 다양한 스케일 생성
python scripts/generate_multiscale_DF2K.py \
    --input datasets/DF2K/DF2K_HR \
    --output datasets/DF2K/DF2K_multiscale
```

### 2단계: 서브이미지 추출
```bash
# GT 이미지 패치 추출
python scripts/extract_subimages.py \
    --input datasets/DF2K/DF2K_HR \
    --output datasets/DF2K/DF2K_HR_sub \
    --crop_size 480 --step 240

# LQ 이미지 패치 추출 (필요시)
python scripts/extract_subimages.py \
    --input datasets/DF2K/DF2K_LR \
    --output datasets/DF2K/DF2K_LR_sub \
    --crop_size 120 --step 60
```

### 3단계: 메타정보 생성
```bash
# GT 전용 메타정보
python scripts/generate_meta_info.py \
    --input datasets/DF2K/DF2K_HR_sub datasets/DF2K/DF2K_multiscale \
    --root datasets/DF2K datasets/DF2K \
    --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt

# 페어 데이터 메타정보 (supervised 학습용)
python scripts/generate_meta_info_pairdata.py \
    --input datasets/DF2K/DF2K_HR_sub datasets/DF2K/DF2K_LR_sub \
    --meta_info datasets/DF2K/meta_info/meta_info_pair.txt
```

### 4단계: 모델 변환 (배포용)
```bash
# 학습 완료 후 ONNX 변환
python scripts/pytorch2onnx.py \
    --input experiments/trained_models/net_g_latest.pth \
    --output deployed_models/realesrgan-custom.onnx
```

## 실행 환경 및 요구사항

### 필수 라이브러리:
- **Python 3.7+**
- **OpenCV (cv2)**: 이미지 처리 및 입출력
- **NumPy**: 수치 연산 및 배열 처리
- **PIL (Pillow)**: 고품질 이미지 리샘플링
- **PyTorch**: 딥러닝 모델 처리
- **tqdm**: 진행률 표시
- **basicsr**: Real-ESRGAN의 기본 아키텍처 라이브러리

### 시스템 요구사항:
- **메모리**: 대용량 이미지 처리를 위한 충분한 RAM (8GB+ 권장)
- **저장공간**: 원본 이미지 크기의 3-5배 여유 공간
- **CPU**: 멀티스레딩 지원 프로세서 (멀티코어 권장)

### 성능 최적화 팁:
- `n_thread` 파라미터를 CPU 코어 수에 맞게 조정
- `compression_level`을 낮춰 처리 속도 향상 (디스크 용량 trade-off)
- SSD 사용으로 I/O 성능 향상

## 각 스크립트가 생성하는 파일들의 형식과 용도

### 1. extract_subimages.py 출력
- **형식**: `원본파일명_s001.png`, `원본파일명_s002.png`, ...
- **용도**: 메모리 효율적인 배치 학습, GPU 메모리 최적화
- **특징**: 순차적 인덱스로 패치 관리

### 2. generate_meta_info.py 출력
- **형식**: 텍스트 파일 (.txt)
- **내용**: 각 라인마다 하나의 이미지 상대 경로
- **예시**:
  ```
  DF2K_HR_sub/0001_s001.png
  DF2K_HR_sub/0001_s002.png
  DF2K_multiscale/0001T0.png
  ```
- **용도**: 데이터로더(DataLoader)에서 이미지 목록 참조

### 3. generate_meta_info_pairdata.py 출력
- **형식**: 텍스트 파일 (.txt)
- **내용**: 각 라인마다 GT, LQ 이미지 쌍의 상대 경로 (쉼표로 구분)
- **예시**:
  ```
  DF2K_HR_sub/0001_s001.png, DF2K_LR_sub/0001_s001.png
  DF2K_HR_sub/0001_s002.png, DF2K_LR_sub/0001_s002.png
  ```
- **용도**: Supervised learning용 GT-LQ 이미지 쌍 관리

### 4. generate_multiscale_DF2K.py 출력
- **형식**: PNG 이미지 파일
- **명명 규칙**:
  - `파일명T0.png`: 0.75배 스케일
  - `파일명T1.png`: 0.5배 스케일  
  - `파일명T2.png`: 1/3배 스케일
  - `파일명T3.png`: 최단변 400px 고정
- **용도**: 멀티스케일 학습 데이터, 스케일 다양성 확보

### 5. pytorch2onnx.py 출력
- **형식**: ONNX 모델 파일 (.onnx)
- **특징**: OPSET 버전 11, 그래프 최적화 적용
- **용도**: 
  - 크로스 플랫폼 모델 배포
  - TensorRT, OpenVINO 등 추론 엔진 지원
  - 모바일/엣지 디바이스 배포

## 주요 활용 사례

### 1. 새로운 데이터셋 준비
고품질 이미지 데이터셋을 Real-ESRGAN 학습용으로 변환할 때:
1. 멀티스케일 이미지 생성으로 다양한 해상도 학습 데이터 확보
2. 서브이미지 추출로 메모리 효율적인 배치 학습 환경 구성
3. 메타정보 생성으로 데이터 로딩 최적화

### 2. 모델 배포 준비
학습된 PyTorch 모델을 실제 서비스에 배포할 때:
1. ONNX 변환으로 다양한 추론 프레임워크 지원
2. 모델 크기 최적화 및 추론 속도 향상
3. 클라우드/엣지 환경에서의 호환성 확보

### 3. 실험 환경 구성
다양한 학습 설정을 실험할 때:
1. 다양한 크롭 크기와 스텝으로 패치 추출 실험
2. 멀티스케일 비율 조정으로 학습 효과 분석
3. 메타정보 구성 변경으로 데이터 로딩 전략 최적화

이 스크립트들은 Real-ESRGAN 모델의 전체 라이프사이클에서 데이터 준비부터 모델 배포까지의 핵심적인 전처리 작업을 담당하며, 효율적인 머신러닝 파이프라인 구축을 지원합니다.