# 가중치 (Weights)

다운로드한 가중치 파일들을 이 폴더에 넣어주세요.

## 모델 설명 (Model Descriptions)

### 일반 이미지용 모델 (General Image Models)
- **RealESRGAN_x4plus.pth**: 일반 사진에 최적화된 4배 업스케일링 모델. 실제 사진의 노이즈 제거와 디테일 복원에 뛰어남
- **RealESRGAN_x2plus.pth**: 일반 사진용 2배 업스케일링 모델. 더 빠른 처리가 필요할 때 사용
- **RealESRNet_x4plus.pth**: GAN 없이 훈련된 기본 모델. 아티팩트가 적지만 디테일 복원 성능은 상대적으로 낮음

### 애니메이션 특화 모델 (Anime-Specialized Models)  
- **RealESRGAN_x4plus_anime_6B.pth**: 애니메이션 이미지에 특화된 4배 업스케일링 모델. 애니메이션 스타일의 선명한 라인과 색상 복원에 최적화

### Real-ESRGAN v3 모델 (Real-ESRGAN v3 Models)
- **realesr-general-x4v3.pth**: 최신 v3 버전의 일반 이미지용 4배 업스케일링 모델. 더 나은 성능과 안정성 제공
- **realesr-general-wdn-x4v3.pth**: 약한 노이즈 제거 기능이 있는 v3 모델. 노이즈가 적은 고품질 이미지에 적합

## 사용 방법 (Usage)
각 모델을 `-n` 또는 `--model_name` 파라미터로 지정하여 사용할 수 있습니다:
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs
python inference_realesrgan.py -n realesr-general-x4v3 -i inputs  
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```
