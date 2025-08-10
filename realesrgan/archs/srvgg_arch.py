#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 컴팩트 VGG 생성자 아키텍처 (SR-VGG Architecture)

이 파일은 Real-ESRGAN에서 사용되는 경량화된 VGG 스타일 초해상도 생성자 네트워크를 구현합니다.
효율적인 추론 속도와 메모리 사용량을 위해 설계된 컴팩트한 구조입니다.

주요 특징:
- VGG 스타일 구조: 연속적인 3x3 합성곱 레이어로 구성
- 컴팩트한 설계: 고해상도 공간에서 연산을 최소화하여 효율성 극대화
- 픽셀 셔플 업샘플링: 서브픽셀 합성곱을 통한 효율적 업스케일링
- 잔차 학습: 입력 이미지와의 잔차를 학습하여 디테일 보존
- 활성화 함수 선택: ReLU, PReLU, LeakyReLU 지원

네트워크 구조:
1. 초기 특징 추출:
   - 첫 번째 Conv2d: 입력 채널 → num_feat 채널 (3x3, stride=1, padding=1)
   - 활성화 함수 적용

2. 특징 처리 본체(Body):
   - num_conv개의 연속적인 합성곱 블록
   - 각 블록: Conv2d(num_feat → num_feat) + 활성화 함수
   - 모든 합성곱: 3x3 커널, stride=1, padding=1

3. 업샘플링 및 출력:
   - 마지막 Conv2d: num_feat → (num_out_ch × upscale²) 채널
   - PixelShuffle: 채널을 공간 차원으로 재배치하여 업스케일링
   - 잔차 연결: 업샘플된 입력 이미지와 합산

픽셀 셔플 업샘플링 (Pixel Shuffle):
- 서브픽셀 합성곱의 핵심 기술
- 채널 차원의 정보를 공간 차원으로 재배치
- 예: upscale=4일 때, (H, W, 16C) → (4H, 4W, C)
- 연산 효율성과 성능을 동시에 만족

활성화 함수 옵션:
- ReLU: 단순하고 빠른 처리, 죽은 뉴런 문제 가능성
- PReLU: 음수 영역에서 학습 가능한 기울기, 더 풍부한 표현력
- LeakyReLU: 음수 영역에서 고정 기울기(0.1), 죽은 뉴런 방지

잔차 학습 (Residual Learning):
- 네트워크가 입력과 출력의 차이(잔차)만 학습
- 입력 이미지의 저주파 정보는 그대로 보존
- 훈련 안정성과 수렴성 향상
- 디테일 정보에 집중한 효과적 학습

설계 철학:
- 효율성 우선: 고해상도 공간에서의 연산 최소화
- 실용성 중시: 실시간 추론 가능한 경량 모델
- 성능 균형: 품질과 속도의 최적 균형점 추구

사용 사례:
- Real-ESRGAN v3 모델 (realesr-general-x4v3.pth)
- 애니메이션 비디오 초해상도 (realesr-animevideov3.pth)
- 실시간 또는 빠른 추론이 필요한 응용 프로그램

성능 특성:
- 빠른 추론 속도: RRDBNet 대비 2-3배 빠름
- 적은 메모리 사용량: 컴팩트한 구조로 GPU 메모리 절약
- 좋은 품질: 효율성 대비 우수한 초해상도 결과

하이퍼파라미터:
- num_conv=16: 일반 이미지용 (더 복잡한 특징 학습)
- num_conv=32: 고품질이 요구되는 경우
- num_feat=64: 기본 특징 채널 수 (메모리와 성능의 균형)

비교 분석:
vs RRDBNet: 더 빠르지만 상대적으로 단순한 구조
vs ESPCN: 픽셀 셔플 사용하지만 더 깊고 복잡한 네트워크
vs SRCNN: VGG 스타일이지만 잔차 학습과 픽셀 셔플 추가

논문 출처: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
개발: Tencent ARC Lab, Xintao Wang et al.
"""

from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F


@ARCH_REGISTRY.register()
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out
