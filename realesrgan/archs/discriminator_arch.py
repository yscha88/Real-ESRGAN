#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 판별자 아키텍처 (Discriminator Architecture)

이 파일은 Real-ESRGAN에서 사용되는 U-Net 기반 판별자 네트워크를 구현합니다.
GAN 훈련에서 생성된 이미지와 실제 이미지를 구별하는 역할을 수행합니다.

주요 특징:
- U-Net 구조: 다운샘플링 후 업샘플링을 통해 다양한 스케일의 특징을 추출
- 스펙트럴 정규화(Spectral Normalization): 훈련 안정성 향상을 위한 가중치 정규화
- 스킵 연결(Skip Connections): U-Net의 핵심 특징으로 세밀한 디테일 보존
- 패치 기반 판별: 입력과 동일한 공간 해상도의 판별 맵 출력

네트워크 구조:
1. 다운샘플링 단계 (Encoder):
   - conv0: 3→64 채널 (stride=1, 초기 특징 추출)
   - conv1: 64→128 채널 (stride=2, 2x 다운샘플링)
   - conv2: 128→256 채널 (stride=2, 4x 다운샘플링)  
   - conv3: 256→512 채널 (stride=2, 8x 다운샘플링)

2. 업샘플링 단계 (Decoder):
   - conv4: 512→256 채널 + 2x 업샘플링 + 스킵 연결(conv2)
   - conv5: 256→128 채널 + 2x 업샘플링 + 스킵 연결(conv1)
   - conv6: 128→64 채널 + 2x 업샘플링 + 스킵 연결(conv0)

3. 최종 판별 레이어:
   - conv7, conv8: 64→64 채널 (추가 특징 정제)
   - conv9: 64→1 채널 (최종 판별 출력)

스펙트럴 정규화(SN) 적용:
- 모든 합성곱 레이어에 적용 (conv0, conv9 제외)
- 가중치의 최대 고유값을 1로 제한하여 훈련 안정성 향상
- 그래디언트 폭발 문제 해결 및 립시츠 연속성 보장

활성화 함수:
- LeakyReLU (negative_slope=0.2): 모든 중간 레이어
- 마지막 레이어는 활성화 함수 없음 (로짓 출력)

입력/출력:
- 입력: (B, C, H, W) 형태의 이미지 텐서
- 출력: (B, 1, H, W) 형태의 판별 맵 (픽셀별 진짜/가짜 확률)

사용 사례:
- Real-ESRGAN 모델의 적대적 손실 계산
- 생성된 고해상도 이미지의 현실성 평가
- 패치 단위의 세밀한 판별을 통한 국소적 품질 향상

성능 특성:
- 메모리 효율적: U-Net 구조로 인한 적은 파라미터 수
- 다중 스케일 판별: 다양한 해상도에서의 특징 활용
- 안정적 훈련: 스펙트럴 정규화를 통한 안정성 보장

논문 출처: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
개발: Tencent ARC Lab, Xintao Wang et al.
"""

from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
