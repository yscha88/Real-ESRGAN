"""
Real-ESRGAN 페어드 데이터셋 (Real-ESRGAN Paired Dataset)
======================================================

이 파일은 Real-ESRGAN 모델의 학습과 평가를 위한 페어드(paired) 데이터셋을 구현합니다.
GT (Ground Truth) 고화질 이미지와 LQ (Low Quality) 저화질 이미지 쌍을 로드하여
supervised learning 방식의 이미지 복원 학습을 지원합니다.

주요 특징 (Key Features):
------------------------
1. 다중 데이터 백엔드 지원 (Multi-backend Data Loading)
   - LMDB: 대용량 데이터셋을 위한 고성능 키-값 데이터베이스
   - Meta Info: 메타 정보 파일 기반 경로 관리
   - Folder: 폴더 스캐닝을 통한 자동 파일 쌍 매칭

2. 고급 데이터 증강 파이프라인 (Advanced Data Augmentation)
   - Paired Random Crop: GT-LQ 쌍에 동일한 위치 크롭 적용
   - 수평 플립과 회전 변환으로 데이터 다양성 증대
   - 스케일링 팩터를 고려한 정확한 공간적 대응 관계 유지

3. 정규화 및 전처리 (Normalization & Preprocessing)
   - 사용자 정의 mean/std 정규화 지원
   - BGR to RGB 색상 공간 변환
   - float32 형식과 [0,1] 범위 정규화

4. 유연한 파일 경로 관리 (Flexible File Path Management)
   - 파일명 템플릿을 통한 유연한 파일 매칭
   - 상대/절대 경로 지원
   - 메타 정보 파일을 통한 커스텀 파일 쌍 정의

데이터 로딩 모드 (Data Loading Modes):
-----------------------------------
1. LMDB 모드: 
   - 고성능 키-값 저장소 활용
   - 대용량 데이터셋에 최적화
   - 빠른 랜덤 액세스 보장

2. Meta Info 모드:
   - 텍스트 파일 기반 경로 관리
   - 커스텀 데이터셋 구성에 유용
   - GT-LQ 파일 쌍 명시적 정의

3. Folder 모드:
   - 폴더 구조 기반 자동 매칭
   - 간단한 데이터셋 구성에 적합
   - 파일명 기반 자동 쌍 매칭

성능 최적화 (Performance Optimizations):
-------------------------------------
- 지연 초기화를 통한 메모리 효율성
- 파일 클라이언트 재사용으로 I/O 오버헤드 최소화
- 텐서 변환 최적화로 GPU 메모리 사용량 최적화
- 정규화 인플레이스 연산으로 메모리 사용량 절약

작성자: Real-ESRGAN Team
용도: 이미지 복원, Super-resolution, Denoising
라이센스: Apache License 2.0
"""

import os
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize


@DATASET_REGISTRY.register()
class RealESRGANPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RealESRGANPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
