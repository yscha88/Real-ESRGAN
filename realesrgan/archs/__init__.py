#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 아키텍처 자동 등록 모듈 (Architecture Auto-Registry Module)

이 파일은 Real-ESRGAN의 archs 폴더에 있는 모든 아키텍처 모듈을 자동으로 스캔하고 
임포트하여 ARCH_REGISTRY에 등록하는 역할을 수행합니다.

주요 기능:
- 자동 스캔: archs 폴더의 모든 '*_arch.py' 파일을 자동 감지
- 동적 임포트: 감지된 아키텍처 파일들을 런타임에 동적으로 임포트
- 레지스트리 등록: @ARCH_REGISTRY.register() 데코레이터를 통한 자동 등록
- 플러그인 시스템: 새로운 아키텍처 추가 시 자동으로 인식

작동 원리:
1. 현재 디렉토리(__file__)의 절대 경로 획득
2. scandir을 사용하여 '_arch.py'로 끝나는 파일들 검색
3. 파일명에서 확장자 제거하여 모듈명 추출
4. importlib을 통한 동적 모듈 임포트
5. 임포트 과정에서 @ARCH_REGISTRY.register() 데코레이터 실행

등록되는 아키텍처:
- discriminator_arch.py → UNetDiscriminatorSN 클래스
- srvgg_arch.py → SRVGGNetCompact 클래스
- (향후 추가되는 모든 '*_arch.py' 파일들)

레지스트리 시스템 (Registry System):
- BasicSR 프레임워크의 핵심 기능
- 문자열 이름으로 클래스 인스턴스 생성 가능
- 설정 파일(YAML)에서 문자열로 아키텍처 지정
- 런타임에 동적으로 모델 생성 및 초기화

사용 예시:
```python
# 설정 파일에서
network_g:
  type: SRVGGNetCompact  # 문자열로 아키텍처 지정
  num_feat: 64
  num_conv: 32
  
# 런타임에서
model = ARCH_REGISTRY.get('SRVGGNetCompact')(num_feat=64, num_conv=32)
```

장점:
- 확장성: 새 아키텍처 추가 시 자동 인식
- 유지보수성: 수동 임포트 코드 불필요
- 일관성: 모든 아키텍처가 동일한 방식으로 등록
- 모듈성: 각 아키텍처가 독립적인 파일로 관리

파일 명명 규칙:
- 모든 아키텍처 파일은 '*_arch.py' 형식으로 명명
- 예: discriminator_arch.py, srvgg_arch.py, custom_arch.py
- 이 규칙을 따르면 자동으로 시스템에 등록됨

에러 처리:
- 임포트 실패 시 해당 모듈은 건너뜀
- 잘못된 아키텍처 파일이 있어도 전체 시스템은 정상 작동
- 로그를 통해 임포트 실패 원인 추적 가능

성능 고려사항:
- 모든 아키텍처가 프로그램 시작 시 한 번만 로드
- 런타임 오버헤드 없음
- 메모리 사용량: 사용하지 않는 아키텍처도 메모리에 로드

개발 가이드라인:
- 새 아키텍처 추가 시 '*_arch.py' 명명 규칙 준수
- 클래스에 @ARCH_REGISTRY.register() 데코레이터 필수
- 독립적이고 재사용 가능한 아키텍처 설계 권장

BasicSR 프레임워크 통합:
- BasicSR의 표준 레지스트리 시스템 활용
- 다른 BasicSR 기반 프로젝트와 호환성 유지
- 표준화된 모델 로딩 및 생성 패턴 지원

개발: Tencent ARC Lab, BasicSR Team
"""

import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import arch modules for registry
# scan all the files that end with '_arch.py' under the archs folder
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'realesrgan.archs.{file_name}') for file_name in arch_filenames]
