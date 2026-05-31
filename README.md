# mute-commentary

영상에서 특정 소리(해설, 목소리 등)를 제거하는 도구. SAM-Audio 모델 사용.

## 스크립트

| 스크립트 | 플랫폼 | 모델 |
|---|---|---|
| `process_onnx.py` | Linux (CUDA) | `matbee/sam-audio-large-onnx` fp16 |
| `process_mlx.py` | macOS (Apple Silicon) | `mlx-community/sam-audio-large-fp16` |
| `process.py` | 범용 (CUDA / MPS / CPU) | `facebook/sam-audio-large` PyTorch |

---

## 설치

### macOS (Apple Silicon)

```bash
conda env create -f environment.yml
conda activate mute-commentary
pip install git+https://github.com/facebookresearch/sam-audio.git
```

> `environment.yml`의 `prefix`가 자신의 경로와 다를 경우 수정하거나 `--prefix` 플래그 사용.

---

### Linux (CUDA)

#### 1. conda 환경 생성

`environment.yml`의 pip 문법 이슈로 직접 생성:

```bash
conda create -n mute-commentary python=3.11 -y
conda activate mute-commentary
```

#### 2. 패키지 설치

mlx 패키지(Apple Silicon 전용) 제외 후 설치:

```bash
grep -v "^mlx" requirements.txt | grep -v "^packaging @" | pip install -r /dev/stdin
pip install git+https://github.com/facebookresearch/sam-audio.git
pip install onnxruntime-gpu soundfile
pip install nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

#### 3. FFmpeg 설치

```bash
conda install -c conda-forge ffmpeg -y
```

#### 4. cuDNN 설치 (onnxruntime-gpu CUDA 지원용)

```bash
conda install -c conda-forge cudnn=9 -y
```

#### 5. CUDA 라이브러리 심링크 생성

onnxruntime-gpu는 CUDA 12 SONAME을 요구하나 torch는 CUDA 13을 설치함. 아래 스크립트로 심링크 생성:

```bash
ENVLIB=$(python -c "import sys; print(sys.prefix)")/lib
PIPLIB=$(python -c "import site; print(site.getsitepackages()[0])")

# cublas (13 -> 12 alias)
ln -sf $ENVLIB/libcublas.so.13   $ENVLIB/libcublas.so.12
ln -sf $ENVLIB/libcublasLt.so.13 $ENVLIB/libcublasLt.so.12

# cuda 12 전용 패키지에서 심링크
ln -sf $PIPLIB/nvidia/cufft/lib/libcufft.so.11         $ENVLIB/libcufft.so.11
ln -sf $PIPLIB/nvidia/curand/lib/libcurand.so.10        $ENVLIB/libcurand.so.10
ln -sf $PIPLIB/nvidia/cuda_runtime/lib/libcudart.so.12  $ENVLIB/libcudart.so.12
```

#### 6. ONNX 모델 다운로드

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'matbee/sam-audio-large-onnx',
    local_dir='onnx_models',
    allow_patterns=['fp16/*'],
)
"
```

---

## 사용법

### Linux (ONNX fp16, 권장)

```bash
conda activate mute-commentary

CUDA_VISIBLE_DEVICES=1 python process_onnx.py 영상.mp4 \
  --prompt "person talking near the camera"
```

**옵션:**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--prompt` / `-p` | `"person talking near the camera"` | 제거할 소리 설명 |
| `--steps` / `-s` | `16` | ODE 스텝 수 (낮을수록 빠름, 최소 8 권장) |
| `--chunk-seconds` | `30.0` | DiT 청킹 단위 (초). 긴 영상 자동 청킹됨 |
| `--overlap-seconds` | `2.0` | 청크 간 crossfade 길이 (초) |
| `--output-dir` / `-o` | 입력 파일과 동일 | 출력 디렉토리 |
| `--device` / `-d` | `cuda` | `cuda` 또는 `cpu` |
| `--keep-target` | — | 잔여음 대신 분리된 소리 출력 |

> GPU 메모리 부족 시 `--chunk-seconds 15` 또는 `CUDA_VISIBLE_DEVICES`로 여유 GPU 지정.

### macOS (MLX, Apple Silicon)

```bash
conda activate mute-commentary
python process_mlx.py 영상.mp4 --prompt "person talking near the camera"
```

### 범용 (PyTorch)

```bash
conda activate mute-commentary
python process.py 영상.mp4 --prompt "person talking near the camera" --reranking 8
```

---

## 출력

입력 파일과 같은 위치에 `{파일명}_muted.{확장자}` 생성.  
`--keep-target` 사용 시 `{파일명}_target.{확장자}` 생성.
