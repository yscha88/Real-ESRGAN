"""
Real-ESRGAN 데이터 모듈 초기화 파일 (Data Module Initialization File)
==================================================================

이 파일은 Real-ESRGAN 프로젝트의 데이터 로딩 모듈을 자동으로 초기화하고 등록하는 역할을 합니다.

주요 기능 (Main Functions):
-------------------------
1. 자동 데이터셋 모듈 검색 (Automatic Dataset Module Discovery)
   - data 폴더 내의 모든 '_dataset.py'로 끝나는 파일을 자동 검색
   - 동적 모듈 임포트를 통한 데이터셋 클래스 자동 등록

2. 레지스트리 기반 데이터셋 관리 (Registry-based Dataset Management)
   - BasicSR의 DATASET_REGISTRY를 활용한 데이터셋 클래스 등록
   - 설정 파일에서 데이터셋 이름으로 동적 인스턴스 생성 지원

포함된 데이터셋 클래스 (Included Dataset Classes):
------------------------------------------------
- RealESRGANDataset: 실시간 열화 생성 데이터셋
- RealESRGANPairedDataset: GT-LQ 쌍 데이터셋

기술적 구현 (Technical Implementation):
------------------------------------
- importlib을 사용한 동적 모듈 임포트
- scandir을 통한 효율적인 파일 시스템 스캔
- 모듈 네임스페이스 자동 관리

작성자: Real-ESRGAN Team
라이센스: Apache License 2.0
"""

import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import dataset modules for registry
# scan all the files that end with '_dataset.py' under the data folder
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'realesrgan.data.{file_name}') for file_name in dataset_filenames]
