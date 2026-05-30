#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str):
    print("모델 로딩 중 (facebook/sam-audio-large)...")
    model = SAMAudio.from_pretrained("facebook/sam-audio-large").to(device).eval()
    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
    return model, processor


def extract_audio(video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )


def remove_voice(
    audio_path: Path,
    residual_path: Path,
    model,
    processor,
    prompt: str,
    device: str,
    reranking: int,
) -> None:
    inputs = processor(
        audios=[str(audio_path)],
        descriptions=[prompt],
    ).to(device)

    with torch.inference_mode():
        result = model.separate(inputs, predict_spans=True, reranking_candidates=reranking)

    # result.residual: 목소리 제거된 나머지 오디오
    residual = result.residual[0].unsqueeze(0).cpu()
    torchaudio.save(str(residual_path), residual, processor.audio_sampling_rate)


def merge(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    model,
    processor,
    prompt: str,
    device: str,
    reranking: int,
) -> Path:
    output_path = output_dir / f"{video_path.stem}_muted{video_path.suffix}"

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        print(f"  [1/3] 오디오 추출...")
        audio_wav = tmp / "audio.wav"
        extract_audio(video_path, audio_wav)

        print(f'  [2/3] 목소리 제거 (프롬프트: "{prompt}")...')
        residual_wav = tmp / "residual.wav"
        remove_voice(audio_wav, residual_wav, model, processor, prompt, device, reranking)

        print(f"  [3/3] 영상 합성...")
        merge(video_path, residual_wav, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="축구 영상에서 매니저 목소리를 제거합니다 (SAM-Audio)."
    )
    parser.add_argument("videos", nargs="+", type=Path, help="처리할 영상 파일 (여러 개 가능)")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="출력 디렉토리 (기본: 입력 파일과 같은 위치)",
    )
    parser.add_argument(
        "--prompt", "-p", default="person talking near the camera",
        help='제거할 소리 설명 (기본: "person talking near the camera")',
    )
    parser.add_argument(
        "--reranking", "-r", type=int, default=1,
        help="품질 향상을 위한 후보 수 (기본: 1, 높을수록 느리고 정확함. 권장: 8)",
    )
    parser.add_argument(
        "--device", "-d", default=None,
        help="연산 장치 (기본: 자동감지 cuda > cpu)",
    )
    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"장치: {device}")

    model, processor = load_model(device)

    for video_path in args.videos:
        video_path = video_path.resolve()
        if not video_path.exists():
            print(f"파일 없음, 건너뜀: {video_path}")
            continue

        output_dir = args.output_dir or video_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n처리 중: {video_path.name}")
        try:
            result = process_video(
                video_path, output_dir, model, processor,
                args.prompt, device, args.reranking,
            )
            print(f"  완료 → {result}")
        except Exception as e:
            print(f"  오류: {e}")


if __name__ == "__main__":
    main()
