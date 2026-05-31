#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path

from mlx_audio.sts import SAMAudio, SAMAudioProcessor, save_audio

MODEL_ID = "mlx-community/sam-audio-large-fp16"


def load_model():
    print(f"모델 로딩 중 ({MODEL_ID})...")
    processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    model = SAMAudio.from_pretrained(MODEL_ID)
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
    chunk_seconds: float,
    overlap_seconds: float,
) -> None:
    batch = processor(
        audios=[str(audio_path)],
        descriptions=[prompt],
    )

    result = model.separate_long(
        batch.audios,
        descriptions=batch.descriptions,
        anchors=batch.anchor_ids,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

    save_audio(result.residual[0], str(residual_path), sample_rate=model.sample_rate)


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
    chunk_seconds: float,
    overlap_seconds: float,
) -> Path:
    output_path = output_dir / f"{video_path.stem}_muted{video_path.suffix}"

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        print(f"  [1/3] 오디오 추출...")
        audio_wav = tmp / "audio.wav"
        extract_audio(video_path, audio_wav)

        print(f'  [2/3] 목소리 제거 (프롬프트: "{prompt}")...')
        residual_wav = tmp / "residual.wav"
        remove_voice(audio_wav, residual_wav, model, processor, prompt, chunk_seconds, overlap_seconds)

        print(f"  [3/3] 영상 합성...")
        merge(video_path, residual_wav, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="축구 영상에서 매니저 목소리를 제거합니다 (SAM-Audio MLX)."
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
        "--chunk-seconds", type=float, default=10.0,
        help="긴 오디오 처리 시 청크 길이 초 (기본: 10.0)",
    )
    parser.add_argument(
        "--overlap-seconds", type=float, default=3.0,
        help="청크 간 겹침 길이 초 (기본: 3.0)",
    )
    args = parser.parse_args()

    model, processor = load_model()

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
                args.prompt, args.chunk_seconds, args.overlap_seconds,
            )
            print(f"  완료 → {result}")
        except Exception as e:
            print(f"  오류: {e}")


if __name__ == "__main__":
    main()
