# Real-ESRGAN 테스트 가이드 (Test Guide)

# 📚 문서 연결성 (Document Hierarchy)

**📍 현재 위치**: L1 - 테스트 시스템 가이드
**🔗 상위 문서**: [L0 - CLAUDE.md](../CLAUDE.md) - Real-ESRGAN 프로젝트 전체 가이드
**📂 테스트 대상 문서**: 
- [L2 - RealESRGANer 유틸리티](../realesrgan/realesrgan_context.md) - 유틸리티 함수 테스트 대상
- [L2 - 아키텍처 분석](../realesrgan/archs/archs_context.md) - 아키텍처 테스트 대상
- [L2 - 데이터 시스템](../realesrgan/data/data_context.md) - 데이터셋 테스트 대상
- [L2 - 모델 시스템](../realesrgan/models/models_context.md) - 모델 테스트 대상

---

## 1. 테스트 구조 개요 (Test Structure Overview)

Real-ESRGAN 프로젝트의 테스트는 다음과 같이 구성되어 있습니다:

```
tests/
├── test_utils.py              # 유틸리티 함수 테스트
├── test_discriminator_arch.py # 판별자 아키텍처 테스트
├── test_dataset.py           # 데이터셋 테스트
├── test_model.py             # 모델 테스트
└── data/                     # 테스트 데이터 및 설정 파일
    ├── gt/                   # Ground Truth 이미지 (고품질)
    ├── lq/                   # Low Quality 이미지 (저품질)
    ├── gt.lmdb/             # GT 이미지 LMDB 데이터베이스
    ├── lq.lmdb/             # LQ 이미지 LMDB 데이터베이스
    ├── *.yml                # 테스트 설정 파일들
    └── meta_info_*.txt      # 메타데이터 파일들
```

## 2. 테스트 파일별 기능 설명 (Test File Functions)

### 2.1 test_utils.py - 유틸리티 테스트 (Utility Tests)

**목적 (Purpose)**: RealESRGANer 클래스의 모든 핵심 기능을 검증합니다.

**주요 검증 내용 (Key Validations)**:
- **모델 초기화 (Model Initialization)**: 기본 모델과 사용자 정의 모델 로딩
- **전처리 (Pre-processing)**: 입력 이미지 크기 조정 및 패딩
- **추론 처리 (Inference Processing)**: 모델을 통한 이미지 처리
- **후처리 (Post-processing)**: 출력 이미지 크기 조정 및 정규화
- **타일 처리 (Tile Processing)**: 큰 이미지를 작은 타일로 나누어 처리
- **향상 기능 (Enhancement)**: 다양한 이미지 포맷 지원 (RGB, 16-bit, 그레이스케일, RGBA)

**테스트되는 주요 기능들 (Main Features Tested)**:
- `pre_process()`: 이미지 전처리 및 크기 변환
- `process()`: 신경망을 통한 이미지 처리
- `post_process()`: 결과 이미지 후처리
- `tile_process()`: 메모리 효율적인 타일 기반 처리
- `enhance()`: 통합 이미지 향상 파이프라인

### 2.2 test_discriminator_arch.py - 판별자 아키텍처 테스트 (Discriminator Architecture Tests)

**목적 (Purpose)**: UNetDiscriminatorSN 아키텍처의 정상 작동을 확인합니다.

**주요 검증 내용 (Key Validations)**:
- **모델 초기화 (Model Initialization)**: UNet 기반 판별자 생성
- **순전파 (Forward Pass)**: CPU와 GPU 환경에서의 정상 추론
- **출력 형태 검증 (Output Shape Validation)**: 예상된 출력 차원 확인

**테스트 환경 (Test Environments)**:
- CPU 환경에서의 기본 동작
- CUDA 사용 가능 시 GPU 환경에서의 동작

### 2.3 test_dataset.py - 데이터셋 테스트 (Dataset Tests)

**목적 (Purpose)**: 두 가지 주요 데이터셋 클래스의 기능을 검증합니다.

#### 2.3.1 RealESRGANDataset 테스트
- **데이터 로딩 (Data Loading)**: 디스크 및 LMDB 백엔드에서의 이미지 로딩
- **데이터 증강 (Data Augmentation)**: 동적 열화 프로세스 시뮬레이션
- **커널 생성 (Kernel Generation)**: 다양한 블러 커널 생성 검증
- **Sinc 커널 처리 (Sinc Kernel Processing)**: Sinc 필터 적용 확인

#### 2.3.2 RealESRGANPairedDataset 테스트
- **페어 데이터 로딩 (Paired Data Loading)**: GT-LQ 이미지 쌍 로딩
- **정규화 처리 (Normalization)**: 이미지 정규화 기능 검증
- **폴더 기반 로딩 (Folder-based Loading)**: 메타데이터 없이 폴더에서 직접 로딩

**지원하는 백엔드 (Supported Backends)**:
- **Disk Backend**: 일반 파일 시스템에서 이미지 로딩
- **LMDB Backend**: 고성능 LMDB 데이터베이스에서 이미지 로딩

### 2.4 test_model.py - 모델 테스트 (Model Tests)

**목적 (Purpose)**: Real-ESRGAN의 두 가지 핵심 모델을 검증합니다.

#### 2.4.1 RealESRNetModel 테스트
- **모델 구성요소 (Model Components)**: RRDBNet 생성자, L1Loss 손실함수
- **데이터 공급 (Data Feeding)**: 훈련 데이터 입력 및 처리
- **열화 시뮬레이션 (Degradation Simulation)**: 실제적인 이미지 열화 과정
- **검증 과정 (Validation Process)**: 모델 성능 평가

#### 2.4.2 RealESRGANModel 테스트
- **GAN 구성요소 (GAN Components)**: 생성자(Generator) + 판별자(Discriminator)
- **손실 함수들 (Loss Functions)**:
  - Pixel Loss (L1Loss): 픽셀 단위 차이 계산
  - Perceptual Loss: 지각적 품질 손실
  - GAN Loss: 적대적 훈련 손실
- **최적화 과정 (Optimization Process)**: 생성자와 판별자의 교대 훈련
- **로그 기록 (Logging)**: 훈련 과정의 각종 손실값 기록

## 3. 테스트 데이터 구성 (Test Data Configuration)

### 3.1 이미지 데이터 (Image Data)
- **GT 폴더 (Ground Truth)**: `tests/data/gt/`
  - `baboon.png`: 480×500×3 픽셀의 컬러 이미지
  - `comic.png`: 360×240×3 픽셀의 컬러 이미지
  
- **LQ 폴더 (Low Quality)**: `tests/data/lq/`
  - GT 이미지에 대응하는 저품질 버전

### 3.2 LMDB 데이터베이스 (LMDB Database)
- **gt.lmdb/**: 고품질 이미지의 LMDB 형태 저장
- **lq.lmdb/**: 저품질 이미지의 LMDB 형태 저장
- 각 LMDB는 `data.mdb`, `lock.mdb`, `meta_info.txt` 파일을 포함

### 3.3 메타데이터 파일 (Metadata Files)
- **meta_info_gt.txt**: GT 이미지 목록
- **meta_info_pair.txt**: GT-LQ 이미지 쌍 정보
- **meta_info.txt (LMDB)**: 이미지 차원 및 압축 정보

## 4. 설정 파일 분석 (Configuration Files Analysis)

### 4.1 test_realesrgan_dataset.yml
- **데이터셋 타입 (Dataset Type)**: RealESRGANDataset
- **블러 커널 설정 (Blur Kernel Settings)**:
  - 1차 블러: 21×21 크기, 6가지 커널 타입
  - 2차 블러: 추가적인 열화 효과
- **증강 옵션 (Augmentation Options)**: 수평 플립, 회전 없음

### 4.2 test_realesrgan_model.yml  
- **네트워크 구조 (Network Architecture)**:
  - Generator: RRDBNet (4개 특징, 1개 블록)
  - Discriminator: UNetDiscriminatorSN (2개 특징)
- **훈련 설정 (Training Settings)**:
  - Adam 옵티마이저 (lr=1e-4)
  - MultiStepLR 스케줄러
  - 총 40만 이터레이션
- **손실 함수 가중치 (Loss Weights)**:
  - Pixel Loss: 1.0
  - Perceptual Loss: 1.0  
  - GAN Loss: 0.1

### 4.3 test_realesrgan_paired_dataset.yml
- **데이터셋 타입 (Dataset Type)**: RealESRGANPairedDataset
- **스케일 (Scale)**: 4배 초해상도
- **이미지 크기 (Image Size)**: GT 128×128, LQ 32×32

### 4.4 test_realesrnet_model.yml
- **RealESRNet 전용 설정 (RealESRNet-specific Config)**
- **단일 생성자 모델 (Generator-only Model)**: 판별자 없음
- **학습률 (Learning Rate)**: 2e-4 (GAN보다 높음)
- **총 훈련 (Total Training)**: 100만 이터레이션

## 5. 테스트 실행 방법 (How to Run Tests)

### 5.1 전체 테스트 실행 (Run All Tests)
```bash
# 현재 디렉토리에서 모든 테스트 실행
python -m pytest tests/

# 상세한 출력과 함께 실행
python -m pytest tests/ -v

# 특정 테스트 파일만 실행
python -m pytest tests/test_utils.py -v
```

### 5.2 개별 테스트 함수 실행 (Run Individual Test Functions)
```bash
# 특정 테스트 함수만 실행
python -m pytest tests/test_utils.py::test_realesrganer -v
python -m pytest tests/test_model.py::test_realesrgan_model -v
```

### 5.3 커버리지 포함 실행 (Run with Coverage)
```bash
# 코드 커버리지와 함께 테스트 실행
python -m pytest tests/ --cov=realesrgan --cov-report=html
```

## 6. 각 테스트의 검증 내용 (What Each Test Validates)

### 6.1 기능적 검증 (Functional Validation)
- **API 호환성 (API Compatibility)**: 모든 공개 메서드가 예상대로 작동
- **입출력 형태 (I/O Shape)**: 입력과 출력 텐서의 차원이 올바름
- **데이터 타입 (Data Types)**: 올바른 데이터 타입 처리
- **에러 핸들링 (Error Handling)**: 잘못된 입력에 대한 적절한 에러 발생

### 6.2 성능적 검증 (Performance Validation)
- **메모리 효율성 (Memory Efficiency)**: 타일 처리를 통한 메모리 사용량 최적화
- **GPU 가속 (GPU Acceleration)**: CUDA 환경에서의 정상 작동
- **배치 처리 (Batch Processing)**: 다중 이미지 동시 처리

### 6.3 품질적 검증 (Quality Validation)
- **이미지 품질 (Image Quality)**: 출력 이미지의 품질 확인
- **색상 공간 (Color Space)**: RGB, 그레이스케일, RGBA 지원
- **비트 깊이 (Bit Depth)**: 8비트, 16비트 이미지 처리

## 7. 테스트 환경 요구사항 (Test Environment Requirements)

### 7.1 필수 의존성 (Required Dependencies)
- Python 3.7+
- PyTorch 1.7+
- BasicSR
- OpenCV-Python
- NumPy
- PyYAML
- pytest

### 7.2 선택적 의존성 (Optional Dependencies)
- CUDA (GPU 가속을 위한)
- pytest-cov (코드 커버리지를 위한)

### 7.3 하드웨어 요구사항 (Hardware Requirements)
- **최소 RAM**: 4GB (CPU 테스트용)
- **권장 RAM**: 8GB+ (GPU 테스트 포함)
- **GPU**: NVIDIA GPU (CUDA 지원 테스트용, 선택사항)

## 8. 테스트 결과 해석 (Interpreting Test Results)

### 8.1 성공적인 테스트 (Successful Tests)
- 모든 `assert` 문이 통과
- 예상된 출력 형태와 값 범위 확인
- 메모리 누수 없음

### 8.2 실패 가능한 원인 (Possible Failure Causes)
- **의존성 누락 (Missing Dependencies)**: 필요한 패키지 미설치
- **CUDA 오류 (CUDA Errors)**: GPU 드라이버 또는 CUDA 버전 문제
- **메모리 부족 (Out of Memory)**: 시스템 메모리 또는 GPU 메모리 부족
- **파일 경로 오류 (File Path Errors)**: 테스트 데이터 파일 누락

### 8.3 디버깅 팁 (Debugging Tips)
- `-s` 플래그로 print 출력 확인: `pytest tests/ -s`
- `--tb=long` 으로 자세한 트레이스백 확인
- 개별 테스트 함수를 단독 실행하여 문제 격리

---

이 테스트 모음은 Real-ESRGAN의 모든 핵심 컴포넌트가 올바르게 작동하는지 확인하여 안정적이고 신뢰할 수 있는 이미지 초해상도 성능을 보장합니다.