"""
Real-ESRGAN 모델 패키지 초기화 모듈

이 모듈은 Real-ESRGAN의 모델 시스템을 관리하는 핵심 초기화 파일입니다.

주요 기능:
1. 모델 모듈 자동 탐지 (Model Module Auto-Discovery)
   - models 폴더 내의 모든 '_model.py'로 끝나는 파일을 자동으로 스캔
   - 동적으로 모델 모듈을 가져와서 MODEL_REGISTRY에 등록

2. 모델 레지스트리 관리 (Model Registry Management)  
   - BasicSR의 MODEL_REGISTRY 시스템과 연동
   - 런타임에 모델 클래스들을 자동으로 등록하여 사용 가능하도록 함

포함된 모델:
- RealESRGANModel: GAN 손실을 사용하는 완전한 Real-ESRGAN 모델
- RealESRNetModel: GAN 손실 없이 MSE 기반으로만 훈련되는 모델

기술적 구현:
- importlib을 사용한 동적 모듈 import
- scandir을 통한 효율적인 파일 스캔
- 확장 가능한 모듈 구조로 새로운 모델 추가 시 자동 인식

사용법:
이 모듈은 realesrgan 패키지가 import될 때 자동으로 실행되어
모든 모델 클래스들을 사용 가능한 상태로 만듭니다.
"""

import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import model modules for registry
# scan all the files that end with '_model.py' under the model folder
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'realesrgan.models.{file_name}') for file_name in model_filenames]
