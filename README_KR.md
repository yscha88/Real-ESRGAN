<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

## <div align="center"><b><a href="README_KR.md">한국어</a></b></div>

<div align="center">

👀[**데모 영상**](#-데모-영상) **|** 🚩[**업데이트**](#-업데이트) **|** 📦[**설치**](#-설치) **|** 🖥[**사용법**](#-사용법) **|** 🧩[**모델**](#-모델) **|** 💻[**추론**](#-추론) **|** 🏰[**데모**](#-데모) **|** 📜[**라이선스**](#-라이선스)
</div>

---

## 🚀 소개
Real-ESRGAN은 실사 이미지 및 비실사 이미지(애니메이션, 게임 등)를 고품질로 업스케일링하는 이미지 초해상도(Super-Resolution) 알고리즘입니다.  
이 프로젝트는 [ESRGAN](https://github.com/xinntao/ESRGAN)을 기반으로 개선되었으며, 더욱 다양한 환경에서 안정적으로 동작하도록 훈련되었습니다.

---

## 📽 데모 영상
[데모 영상 바로가기](https://github.com/xinntao/Real-ESRGAN#-demos-videos)

---

## 🚩 업데이트
- 2021.09: Real-ESRGAN v0.2.5 릴리즈
- 2021.08: 초기 릴리즈

---

## 📦 설치
```bash
git clone git@github.com:yscha88/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
```

---

## 🖥 사용법
### 명령어 예시
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i input.jpg
```

옵션:
- `-n`: 모델 이름 (예: `RealESRGAN_x4plus`)
- `-i`: 입력 이미지 경로 (폴더 또는 파일)
- `-o`: 출력 폴더
- `--outscale`: 업스케일 비율
- `--face_enhance`: 얼굴 복원 옵션

---

## 🧩 모델
- **RealESRGAN_x4plus**: 일반 이미지에 최적화
- **RealESRGAN_x4plus_anime_6B**: 애니메이션/만화에 최적화

모델 다운로드는 [여기](https://github.com/xinntao/Real-ESRGAN#-models) 참고.

---

## 💻 추론
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i input.jpg --outscale 4
```

## 📜 라이선스
이 프로젝트는 [Apache 2.0 License](LICENSE)에 따라 배포됩니다.
