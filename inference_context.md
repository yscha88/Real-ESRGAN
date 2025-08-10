# Real-ESRGAN 추론(Inference) 시스템 가이드

# 📚 문서 연결성 (Document Hierarchy)

**📍 현재 위치**: L1 - 추론 시스템 가이드
**🔗 상위 문서**: [L0 - CLAUDE.md](./CLAUDE.md) - Real-ESRGAN 프로젝트 전체 가이드
**📂 하위 문서**: 
- [L2 - RealESRGANer 유틸리티](./realesrgan/realesrgan_context.md) - 핵심 추론 클래스 및 유틸리티
- [L2 - 아키텍처 분석](./realesrgan/archs/archs_context.md) - 추론에 사용되는 신경망 구조

---

## 1. 추론 스크립트 개요

Real-ESRGAN은 실제 이미지에서의 초해상도(Super-Resolution) 품질을 향상시키기 위해 개발된 GAN 기반 모델입니다. 이 프로젝트는 두 가지 주요 추론 스크립트를 제공하여 다양한 입력 형태에 대응합니다:

- **inference_realesrgan.py**: 정적 이미지 초해상도 처리 전용
- **inference_realesrgan_video.py**: 비디오 및 애니메이션 초해상도 처리 전용

두 스크립트 모두 동일한 Real-ESRGAN 모델을 기반으로 하지만, 입력 데이터의 특성과 처리 방식에서 차이를 보입니다.

## 2. 각 파일별 상세 분석

### 2.1 inference_realesrgan.py - 이미지 초해상도 처리

#### 주요 기능
- 단일 이미지 또는 폴더 내 모든 이미지 배치 처리
- 6가지 사전 훈련된 모델 지원
- GFPGAN을 활용한 얼굴 향상(Face Enhancement) 기능
- RGBA 투명도 채널 지원
- 자동 모델 가중치 다운로드
- GPU 메모리 절약을 위한 타일(Tile) 처리

#### 처리 과정
1. **입력 검증**: 단일 파일 또는 폴더 경로 확인
2. **모델 초기화**: 지정된 모델에 따른 아키텍처 구성
3. **모델 로드**: 로컬 또는 원격에서 가중치 파일 로드
4. **이미지 처리**: 순차적 이미지 처리 및 업스케일링
5. **결과 저장**: 지정된 출력 형식으로 저장

#### 지원 입력 형식
- 이미지 파일: JPEG, PNG, BMP, TIFF 등
- 폴더: 내부 모든 이미지 파일 자동 처리
- RGBA 이미지: 투명도 채널 보존

### 2.2 inference_realesrgan_video.py - 비디오 초해상도 처리

#### 주요 기능
- 비디오 파일 프레임 단위 처리
- 멀티프로세싱 및 GPU 병렬 처리 지원
- FFmpeg 기반 비디오 스트림 처리
- 오디오 트랙 보존
- FLV → MP4 자동 변환
- 프레임 추출 모드 지원

#### 처리 과정
1. **비디오 메타데이터 추출**: 해상도, FPS, 오디오 정보 수집
2. **멀티프로세싱 분할**: GPU 수에 따른 비디오 분할
3. **프레임별 처리**: 실시간 스트림 처리 또는 추출된 프레임 처리
4. **결과 병합**: 처리된 세그먼트를 하나의 비디오로 결합
5. **오디오 병합**: 원본 오디오 트랙과 결합

#### 지원 입력 형식
- 비디오 파일: MP4, AVI, MOV, FLV 등
- 이미지 시퀀스: 연속된 프레임 이미지 폴더
- 스트림 형식: FFmpeg이 지원하는 모든 형식

## 3. 기능 비교 분석

### 3.1 이미지 vs 비디오 처리의 핵심 차이점

| 특성 | 이미지 처리 | 비디오 처리 |
|------|-------------|-------------|
| **메모리 사용** | 이미지별 독립적 | 스트리밍 처리로 메모리 효율적 |
| **처리 속도** | 개별 최적화 | 멀티프로세싱 병렬 처리 |
| **품질 일관성** | 이미지별 최적 품질 | 프레임 간 일관성 유지 |
| **리소스 관리** | 단순한 배치 처리 | 복잡한 스트림 관리 |
| **오디오 처리** | 해당 없음 | 원본 오디오 보존 |

### 3.2 공통 기능
- 동일한 6가지 모델 지원
- GFPGAN 얼굴 향상 기능
- 타일 처리를 통한 메모리 최적화
- FP16/FP32 정밀도 선택
- 노이즈 제거 강도 조절 (v3 모델)

## 4. 지원 모델 분석

### 4.1 모델 분류 및 특성

| 모델명 | 아키텍처 | 배율 | 블록 수 | 특화 용도 | 메모리 사용량 |
|--------|----------|------|---------|-----------|---------------|
| **RealESRGAN_x4plus** | RRDBNet | 4x | 23 | 일반 이미지 | 높음 |
| **RealESRNet_x4plus** | RRDBNet | 4x | 23 | GAN 없는 업스케일 | 높음 |
| **RealESRGAN_x4plus_anime_6B** | RRDBNet | 4x | 6 | 애니메이션 | 중간 |
| **RealESRGAN_x2plus** | RRDBNet | 2x | 23 | 2배 업스케일 | 중간 |
| **realesr-animevideov3** | SRVGGNet | 4x | 16conv | 애니메이션 비디오 | 낮음 |
| **realesr-general-x4v3** | SRVGGNet | 4x | 32conv | 일반용 v3 | 낮음 |

### 4.2 모델 선택 가이드
- **사진/실사**: RealESRGAN_x4plus 또는 realesr-general-x4v3
- **애니메이션/일러스트**: RealESRGAN_x4plus_anime_6B 또는 realesr-animevideov3
- **메모리 제한**: SRVGGNet 기반 모델 (animevideov3, general-x4v3)
- **빠른 처리**: RealESRNet_x4plus (GAN 없음)

## 5. 매개변수 상세 분석

### 5.1 공통 매개변수

#### 필수 매개변수
```bash
-i, --input          # 입력 경로 (파일 또는 폴더)
-n, --model_name     # 사용할 모델명
-o, --output         # 출력 폴더 경로
```

#### 품질 관련 매개변수
```bash
-s, --outscale       # 최종 업스케일 배율 (기본: 4)
-dn, --denoise_strength  # 노이즈 제거 강도 0~1 (v3 모델 전용)
--face_enhance       # GFPGAN 얼굴 향상 사용
--fp32               # FP32 정밀도 사용 (기본: FP16)
```

#### 메모리 최적화 매개변수
```bash
-t, --tile          # 타일 크기 (0=비활성화, 권장: 400-800)
--tile_pad          # 타일 패딩 크기 (기본: 10)
--pre_pad           # 테두리 사전 패딩 (기본: 0)
```

#### 출력 제어 매개변수
```bash
--suffix            # 출력 파일 접미사 (기본: 'out')
--ext               # 출력 확장자 (auto/jpg/png)
--alpha_upsampler   # 투명도 채널 업샘플러 (realesrgan/bicubic)
-g, --gpu-id        # 사용할 GPU ID
```

### 5.2 비디오 전용 매개변수

#### 비디오 처리 매개변수
```bash
--fps               # 출력 비디오 FPS 지정
--ffmpeg_bin        # FFmpeg 실행 파일 경로
--extract_frame_first  # 프레임 사전 추출 모드
--num_process_per_gpu  # GPU당 프로세스 수 (기본: 1)
```

## 6. 사용 예시

### 6.1 이미지 처리 예시

#### 기본 사용법
```bash
# 단일 이미지 처리
python inference_realesrgan.py -n RealESRGAN_x4plus -i input.jpg -o results

# 폴더 내 모든 이미지 처리
python inference_realesrgan.py -n RealESRGAN_x4plus -i input_folder -o results
```

#### 고급 사용법
```bash
# 애니메이션 이미지 + 얼굴 향상 + 타일 처리
python inference_realesrgan.py \
  -n RealESRGAN_x4plus_anime_6B \
  -i anime_images \
  -o results \
  --face_enhance \
  -t 400 \
  --suffix enhanced

# v3 모델 + 노이즈 제거 + FP32 정밀도
python inference_realesrgan.py \
  -n realesr-general-x4v3 \
  -i noisy_images \
  -o results \
  -dn 0.8 \
  --fp32 \
  -s 2
```

### 6.2 비디오 처리 예시

#### 기본 비디오 처리
```bash
# 애니메이션 비디오 처리
python inference_realesrgan_video.py \
  -n realesr-animevideov3 \
  -i input_video.mp4 \
  -o results

# 일반 비디오 처리 with 얼굴 향상
python inference_realesrgan_video.py \
  -n RealESRGAN_x4plus \
  -i movie.mp4 \
  -o results \
  --face_enhance
```

#### 고성능 처리 설정
```bash
# 멀티GPU + 멀티프로세싱
python inference_realesrgan_video.py \
  -n realesr-animevideov3 \
  -i long_video.mp4 \
  -o results \
  --num_process_per_gpu 2 \
  -t 200 \
  --fps 60

# 프레임 추출 모드 (고품질)
python inference_realesrgan_video.py \
  -n RealESRGAN_x4plus_anime_6B \
  -i anime_video.mp4 \
  -o results \
  --extract_frame_first \
  --fp32
```

### 6.3 특수 상황별 사용법

#### RGBA 이미지 처리
```bash
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i transparent_image.png \
  -o results \
  --alpha_upsampler realesrgan \
  --ext png
```

#### 메모리 제한 환경
```bash
# 작은 타일 크기 + FP32 비활성화
python inference_realesrgan.py \
  -n realesr-general-x4v3 \
  -i large_images \
  -o results \
  -t 200 \
  --tile_pad 5
```

## 7. 성능 최적화 가이드

### 7.1 메모리 최적화

#### 타일 크기 최적화
- **8GB GPU**: --tile 400-600
- **6GB GPU**: --tile 300-400  
- **4GB GPU**: --tile 200-300
- **2GB GPU**: --tile 100-200

#### 모델 선택 최적화
- **메모리 우선**: SRVGGNet 기반 모델 (animevideov3, general-x4v3)
- **품질 우선**: RRDBNet 기반 모델 (x4plus 시리즈)

### 7.2 처리 속도 최적화

#### 이미지 처리 최적화
```bash
# FP16 사용 + 적정 타일 크기
python inference_realesrgan.py \
  -n realesr-general-x4v3 \
  -i images \
  -o results \
  -t 600 \
  --tile_pad 10
```

#### 비디오 처리 최적화
```bash
# 멀티프로세싱 + 스트림 처리
python inference_realesrgan_video.py \
  -n realesr-animevideov3 \
  -i video.mp4 \
  -o results \
  --num_process_per_gpu 2 \
  -t 400
```

### 7.3 품질 최적화

#### 최고 품질 설정
```bash
# FP32 + 큰 타일 + 얼굴 향상
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i input.jpg \
  -o results \
  --fp32 \
  -t 0 \
  --face_enhance \
  --pre_pad 20
```

#### 노이즈 제거 최적화 (v3 모델)
```bash
# 노이즈가 많은 이미지: 강한 노이즈 제거
python inference_realesrgan.py \
  -n realesr-general-x4v3 \
  -i noisy_image.jpg \
  -o results \
  -dn 0.9

# 디테일 보존: 약한 노이즈 제거
python inference_realesrgan.py \
  -n realesr-general-x4v3 \
  -i detailed_image.jpg \
  -o results \
  -dn 0.2
```

## 8. 문제 해결 가이드

### 8.1 일반적인 오류 및 해결책

#### CUDA Out of Memory
```bash
# 오류 메시지: "CUDA out of memory"
# 해결책: 타일 크기 축소
python inference_realesrgan.py -n RealESRGAN_x4plus -i input.jpg -t 200

# 또는 더 작은 모델 사용
python inference_realesrgan.py -n realesr-general-x4v3 -i input.jpg
```

#### 모델 다운로드 실패
```bash
# 수동 다운로드 후 weights 폴더에 배치
# 또는 --model_path로 직접 경로 지정
python inference_realesrgan.py \
  --model_path /path/to/RealESRGAN_x4plus.pth \
  -i input.jpg
```

#### FFmpeg 관련 오류 (비디오)
```bash
# FFmpeg 경로 지정
python inference_realesrgan_video.py \
  -n realesr-animevideov3 \
  -i video.mp4 \
  --ffmpeg_bin /usr/local/bin/ffmpeg

# Windows에서
python inference_realesrgan_video.py \
  -n realesr-animevideov3 \
  -i video.mp4 \
  --ffmpeg_bin "C:\ffmpeg\bin\ffmpeg.exe"
```

### 8.2 특수 상황 대응

#### 대용량 비디오 처리
```bash
# 프레임 추출 모드 사용
python inference_realesrgan_video.py \
  -n realesr-animevideov3 \
  -i large_video.mp4 \
  -o results \
  --extract_frame_first \
  -t 300
```

#### 투명도 채널 문제
```bash
# RGBA 이미지는 반드시 PNG로 저장
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i transparent.png \
  -o results \
  --ext png \
  --alpha_upsampler realesrgan
```

#### 얼굴 향상 제한
- 애니메이션 모델에서는 자동으로 비활성화됨
- 실사 모델에서만 사용 권장
- GFPGAN 의존성 확인 필요

### 8.3 성능 진단

#### GPU 사용률 모니터링
```bash
# 별도 터미널에서 실행
watch -n 1 nvidia-smi

# 또는 Windows에서
nvidia-smi -l 1
```

#### 메모리 사용량 확인
```bash
# 처리 중 메모리 모니터링
python -c "
import torch
print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
torch.cuda.empty_cache()
"
```

## 9. 확장 가능성 및 커스터마이징

### 9.1 모델 커스터마이징

#### 커스텀 모델 사용
```python
# 자체 훈련된 모델 사용 예시
python inference_realesrgan.py \
  --model_path /path/to/custom_model.pth \
  -n RealESRGAN_x4plus \
  -i input.jpg
```

#### 모델 아키텍처 수정
- `realesrgan/archs/` 폴더의 아키텍처 파일 수정
- 새로운 모델 구조 정의 및 등록
- inference 스크립트에 새 모델 조건 추가

### 9.2 배치 처리 자동화

#### 스크립트 배치 처리
```bash
#!/bin/bash
# batch_process.sh

for model in RealESRGAN_x4plus realesr-general-x4v3 RealESRGAN_x4plus_anime_6B
do
    echo "Processing with $model"
    python inference_realesrgan.py \
      -n $model \
      -i input_folder \
      -o results_$model \
      -t 400
done
```

#### Python 래퍼 스크립트
```python
# custom_inference.py
import os
import subprocess

models = [
    'RealESRGAN_x4plus',
    'realesr-general-x4v3',
    'RealESRGAN_x4plus_anime_6B'
]

for model in models:
    cmd = [
        'python', 'inference_realesrgan.py',
        '-n', model,
        '-i', 'input_folder',
        '-o', f'results_{model}',
        '-t', '400'
    ]
    subprocess.run(cmd)
```

### 9.3 API 통합

#### Flask 웹 서비스 예시
```python
from flask import Flask, request, send_file
import subprocess
import os

app = Flask(__name__)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    # 파일 업로드 처리
    file = request.files['image']
    model = request.form.get('model', 'RealESRGAN_x4plus')
    
    # 임시 저장
    input_path = f'temp/input_{file.filename}'
    file.save(input_path)
    
    # Real-ESRGAN 실행
    cmd = [
        'python', 'inference_realesrgan.py',
        '-n', model,
        '-i', input_path,
        '-o', 'temp/output'
    ]
    subprocess.run(cmd)
    
    # 결과 반환
    output_path = f'temp/output/{file.filename.split(".")[0]}_out.jpg'
    return send_file(output_path)

if __name__ == '__main__':
    app.run(debug=True)
```

### 9.4 성능 프로파일링

#### 처리 시간 측정
```python
import time
import argparse

def timed_inference(args):
    start_time = time.time()
    
    # 원본 inference 코드 실행
    # ... 추론 로직 ...
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"처리 시간: {processing_time:.2f}초")
    print(f"초당 프레임: {total_frames/processing_time:.2f}")
    
    return processing_time
```

#### 메모리 사용량 로깅
```python
import torch
import psutil

def log_memory_usage():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 메모리 사용: {gpu_memory:.2f}GB, 캐시: {gpu_cached:.2f}GB")
    
    cpu_memory = psutil.virtual_memory().percent
    print(f"CPU 메모리 사용률: {cpu_memory:.1f}%")
```

## 10. 모범 사례 (Best Practices)

### 10.1 프로덕션 환경 권장사항

#### 안정성 확보
- 예외 처리 강화 및 오류 로깅
- 입력 파일 검증 및 형식 확인
- 출력 디렉토리 권한 확인
- 충분한 디스크 공간 확인

#### 성능 최적화
- GPU 메모리에 맞는 타일 크기 설정
- 배치 크기에 따른 멀티프로세싱 조절
- 모델별 최적 매개변수 프로파일 생성

#### 리소스 관리
- 처리 후 GPU 메모리 정리
- 임시 파일 자동 삭제
- 프로세스 풀 적절한 관리

### 10.2 품질 관리

#### 입력별 모델 선택
- **사진/실사**: RealESRGAN_x4plus 또는 realesr-general-x4v3
- **애니메이션**: RealESRGAN_x4plus_anime_6B 또는 realesr-animevideov3
- **저해상도 옛 사진**: face_enhance 옵션 활용
- **노이즈가 많은 이미지**: v3 모델 + denoise_strength 조절

#### 결과 검증
- 출력 이미지 품질 자동 검사
- 원본 대비 개선도 측정
- 아티팩트 발생 여부 확인

이 가이드를 통해 Real-ESRGAN 추론 시스템을 효과적으로 활용하여 다양한 이미지 및 비디오 초해상도 작업을 수행할 수 있습니다. 각 매개변수와 모델의 특성을 이해하고 상황에 맞는 최적 설정을 선택하는 것이 최고의 결과를 얻는 핵심입니다.