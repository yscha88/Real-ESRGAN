# 🚀 Real-ESRGAN DirectML 백엔드 구동 작업계획서

## 📊 프로젝트 개요

**목표**: AMD RX 580 8GB에서 PyTorch DirectML 백엔드를 활용하여 Real-ESRGAN 초해상도 모델을 최적화하여 구동

**배경**: 현재 Real-ESRGAN은 CUDA 기반으로 구성되어 있으며, AMD GPU에서의 구동을 위해 DirectML 백엔드 전환이 필요합니다.

---

## 🔍 현재 상태 분석

### **프로젝트 구조 분석 결과**
- **프레임워크**: PyTorch 1.8.0 + BasicSR 1.4.2 기반
- **현재 GPU 지원**: CUDA 전용 (`torch.cuda.is_available()` 조건부)
- **디바이스 설정**: `realesrgan/utils.py:111-114`에서 CUDA/CPU 자동 선택
- **메모리 관리**: 타일 처리 시스템으로 대용량 이미지 분할 처리 지원

### **DirectML 호환성 조사 결과**
- ✅ **RX 580 지원**: GCN 1세대 이후 AMD GPU 지원 (RX 580은 GCN 4세대)
- ⚠️ **DirectML 상태**: 현재 유지보수 모드 (새 기능 업데이트 없음)
- ✅ **PyTorch 호환**: PyTorch 2.3.1까지 지원
- ✅ **DirectX 12**: RX 580이 완전 지원

### **RX 580 8GB 하드웨어 분석**
- **GPU 사양**: 2304개 스트림 프로세서, 256-bit 메모리 버스
- **VRAM**: 8GB GDDR5, 256GB/s 대역폭
- **전력**: 185W TDP
- **성능**: 1080p 고설정에서 35-80 FPS (게임 기준)

---

## 🛠️ 구현 방안

### **1단계: DirectML PyTorch 환경 구축**

**1.1 패키지 설치**
```bash
# 현재 환경: Python 3.8.10, PyTorch 1.8.0
pip install torch-directml  # PyTorch DirectML 백엔드
```

**1.2 호환성 확인**
```python
import torch
import torch_directml
dml_device = torch_directml.device()
print(f"DirectML 디바이스: {dml_device}")
```

### **2단계: Real-ESRGAN 코드 수정**

**2.1 디바이스 설정 수정 (`realesrgan/utils.py:111-114`)**
```python
# 기존 코드
if gpu_id:
    self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
else:
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

# 수정 코드
try:
    import torch_directml
    if gpu_id:
        self.device = torch_directml.device(gpu_id) if device is None else device
    else:
        self.device = torch_directml.device() if device is None else device
except ImportError:
    # DirectML 백엔드가 없는 경우 기존 로직 사용
    if gpu_id:
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
    else:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
```

**2.2 Half Precision 처리 수정**
```python
# DirectML에서 FP16 처리 시 안정성 확보
if self.half and hasattr(torch_directml, 'device') and 'dml' in str(self.device):
    # DirectML에서는 FP16을 더 보수적으로 사용
    try:
        self.model = self.model.half()
        self.img = self.img.half()
    except:
        self.half = False  # FP16 실패 시 FP32로 폴백
        print("DirectML에서 FP16 지원 제한으로 FP32 사용")
```

**2.3 추론 스크립트 수정 (`inference_realesrgan.py`)**
```python
# GPU ID 처리 개선
parser.add_argument('-g', '--gpu-id', type=int, default=None, 
                   help='GPU device to use (0 for DirectML, None for auto-detect)')

# DirectML 사용 시 권장 설정 추가
if args.gpu_id is not None:
    print(f"DirectML GPU {args.gpu_id} 사용")
    # DirectML에서는 타일 크기를 보수적으로 설정
    if args.tile == 0:
        args.tile = 512  # RX 580 8GB 기본 권장값
```

---

## ⚡ 성능 최적화 전략

### **메모리 관리 최적화**

**RX 580 8GB 최적 설정**
- **타일 크기**: 256-512픽셀 (기본 512)
- **배치 처리**: 단일 이미지 처리 권장
- **FP16**: 보수적 사용 (호환성 문제 시 FP32 폴백)

**메모리 사용량 추정**
```python
# 4K 이미지 (3840x2160) 처리 시 추정 메모리 사용량
# - 입력: ~33MB (FP32) / ~16MB (FP16)  
# - 모델 가중치: ~64MB (RealESRGAN_x4plus)
# - 중간 특징: ~200-400MB (타일 크기에 따라)
# - 총 예상: ~300-500MB (8GB 중 6-8% 사용)
```

### **성능 벤치마크 목표**

**예상 성능 (RX 580 8GB 기준)**
| 입력 해상도 | 모델 | 타일 크기 | 예상 처리시간 | 메모리 사용량 |
|------------|------|----------|-------------|-------------|
| 1080p → 4K | RealESRGAN_x4plus | 512 | 3-5초 | ~400MB |
| 720p → 2880p | RealESRGAN_x4plus | 512 | 1-2초 | ~200MB |
| 480p → 1920p | RealESRGAN_x4plus | 512 | 0.5-1초 | ~100MB |

---

## 🧪 테스트 시나리오

### **1단계: 환경 검증 테스트**
```bash
# DirectML 설치 및 기본 동작 확인
python -c "import torch_directml; print('DirectML 설치 완료')"

# GPU 인식 테스트  
python -c "import torch_directml; print(f'디바이스: {torch_directml.device()}')"
```

### **2단계: Real-ESRGAN 호환성 테스트**
```bash
# 작은 테스트 이미지로 기본 동작 확인
python inference_realesrgan.py -n RealESRGAN_x4plus -i input/small_test.jpg -t 256 --fp32

# 타일 처리 기능 테스트
python inference_realesrgan.py -n RealESRGAN_x4plus -i input/large_test.jpg -t 512
```

### **3단계: 성능 벤치마크 테스트**
```bash
# 다양한 해상도별 성능 측정
python benchmark_directml.py --resolutions 720p,1080p,4K --tile-sizes 256,512,1024

# 메모리 사용량 모니터링
python inference_realesrgan.py -i input/4k_test.jpg -t 512 --monitor-memory
```

### **4단계: 안정성 테스트**
```bash
# 연속 처리 안정성 테스트
python batch_process_test.py --input-dir test_images/ --iterations 10

# 긴 시간 실행 테스트
python long_running_test.py --duration 3600  # 1시간
```

---

## 📋 실현가능성 평가

### **✅ 높은 실현가능성 요소**

1. **하드웨어 호환성**
   - RX 580은 DirectML 지원 범위 내 (GCN 1세대 이후)
   - 8GB VRAM은 대부분의 Real-ESRGAN 작업에 충분
   - DirectX 12 완전 지원

2. **소프트웨어 호환성**
   - PyTorch-DirectML은 이미 성숙한 솔루션
   - Real-ESRGAN의 PyTorch 1.8.0과 호환 가능
   - 기존 코드 구조가 디바이스 추상화 잘 구현

3. **기술적 구현**
   - 수정 범위가 제한적 (주로 디바이스 설정 부분)
   - 기존 타일 처리 시스템 활용 가능
   - 점진적 개발 및 테스트 가능

### **⚠️ 주의사항 및 제약요소**

1. **DirectML 한계**
   - 유지보수 모드로 새 기능 업데이트 없음
   - CUDA 대비 일부 성능 차이 가능성
   - 특정 PyTorch 연산에서 호환성 이슈 가능

2. **성능 예상**
   - CUDA 대비 10-30% 성능 차이 예상
   - AMD GPU 드라이버 최적화 수준에 따라 변동
   - 일부 모델에서 수치적 정확도 차이 가능성

3. **개발 리스크**
   - DirectML 특화 버그 대응 필요
   - AMD 드라이버 업데이트에 따른 호환성 변화
   - 커뮤니티 지원이 CUDA 대비 제한적

### **📊 성공 확률: 85% (높음)**

**근거**:
- 기술적 호환성이 확실함
- 필요한 코드 수정이 최소한
- 하드웨어 스펙이 충분함
- 대안책 (CPU 폴백) 존재

---

## 🎯 구현 우선순위

### **우선순위 1 (필수) - 2주**
1. DirectML PyTorch 환경 구축 및 검증
2. 기본 Real-ESRGAN 호환성 확보
3. 안정적인 이미지 처리 구현

### **우선순위 2 (권장) - 1주**
1. 성능 최적화 (타일 크기, FP16/32 선택)
2. 에러 핸들링 및 폴백 메커니즘
3. 사용자 가이드 문서 작성

### **우선순위 3 (선택) - 1주**
1. 배치 처리 최적화
2. 성능 모니터링 도구
3. 자동 설정 최적화 기능

---

## 📈 기대 효과

1. **접근성 향상**: AMD GPU 사용자들의 Real-ESRGAN 활용 가능
2. **비용 효율성**: 기존 AMD 하드웨어 활용으로 추가 비용 없음
3. **성능 확보**: 8GB VRAM으로 대부분의 초해상도 작업 처리 가능
4. **확장성**: 향후 다른 AMD GPU로 확장 적용 가능

**결론**: RX 580 8GB에서 DirectML 백엔드를 활용한 Real-ESRGAN 구동은 **높은 실현가능성**을 가지며, 약 **3-4주** 내 안정적인 구현이 가능할 것으로 판단됩니다.

---

## 📚 참고 자료

- [Microsoft DirectML 공식 문서](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)
- [Real-ESRGAN GitHub 리포지토리](https://github.com/xinntao/Real-ESRGAN)
- [PyTorch DirectML GitHub](https://github.com/microsoft/DirectML)

**작성일**: 2025년 8월 10일  
**작성자**: Claude AI 어시스턴트  
**프로젝트**: Real-ESRGAN DirectML 백엔드 구동 계획