# NCNN 모델로 변환하는 방법

1. `scripts/pytorch2onnx.py`를 사용하여 onnx 모델로 변환합니다. 해당 코드를 적절히 수정해야 합니다.
1. onnx 모델을 ncnn 모델로 변환합니다.
    1. `cd ncnn-master\ncnn\build\tools\onnx`
    1. `onnx2ncnn.exe realesrgan-x4.onnx realesrgan-x4-raw.param realesrgan-x4-raw.bin`
1. ncnn 모델을 최적화합니다.
    1. fp16 모드
        1. `cd ncnn-master\ncnn\build\tools`
        1. `ncnnoptimize.exe realesrgan-x4-raw.param realesrgan-x4-raw.bin realesrgan-x4.param realesrgan-x4.bin 1`
1. `realesrgan-x4.param`에서 blob 이름을 수정합니다: `data`와 `output`
