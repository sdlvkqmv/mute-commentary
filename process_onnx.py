#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soundfile as sf
import torchaudio
import torch


DEFAULT_MODEL_DIR = Path(__file__).parent / "onnx_models" / "fp16"


class T5ONNXTokenizer:
    def __init__(self, sp_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)

    def encode(self, text: str) -> np.ndarray:
        ids = self.sp.encode(text)
        if len(ids) > 0 and ids[-1] != 1:
            ids.append(1)
        elif len(ids) == 0:
            ids = [1]
        return np.array(ids, dtype=np.int64).reshape(1, -1)


class SAMAudioONNXPipeline:
    def __init__(self, model_dir: str, device: str = "cuda", num_ode_steps: int = 16):
        self.num_ode_steps = num_ode_steps
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        print("ONNX 모델 로딩 중...")
        self.dacvae_encoder = ort.InferenceSession(
            os.path.join(model_dir, "dacvae_encoder.onnx"), providers=providers
        )
        self.dacvae_decoder = ort.InferenceSession(
            os.path.join(model_dir, "dacvae_decoder.onnx"), providers=providers
        )
        self.t5_encoder = ort.InferenceSession(
            os.path.join(model_dir, "t5_encoder.onnx"), providers=providers
        )
        self.dit = ort.InferenceSession(
            os.path.join(model_dir, "dit_single_step.onnx"), providers=providers
        )
        print("  모든 모델 로드 완료")

        sp_path = os.path.join(model_dir, "tokenizer", "spiece.model")
        self.tokenizer = T5ONNXTokenizer(sp_path)

        first_input = self.dit.get_inputs()[0]
        self._fp16 = first_input.type == "tensor(float16)"
        self._fdtype = np.float16 if self._fp16 else np.float32

    def _encode_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)
        elif audio.ndim == 2:
            audio = audio.reshape(1, *audio.shape)
        return self.dacvae_encoder.run(
            ["latent_features"], {"audio": audio.astype(np.float32)}
        )[0]

    def _decode_audio(self, latent: np.ndarray) -> np.ndarray:
        chunk_size = 25
        hop_length = 1920
        _, _, time_steps = latent.shape
        chunks = []
        for start in range(0, time_steps, chunk_size):
            end = min(start + chunk_size, time_steps)
            chunk = latent[:, :, start:end]
            actual = chunk.shape[2]
            if actual < chunk_size:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, chunk_size - actual)))
            out = self.dacvae_decoder.run(
                ["waveform"], {"latent_features": chunk.astype(np.float32)}
            )[0]
            if actual < chunk_size:
                out = out[:, :, : actual * hop_length]
            chunks.append(out)
        return np.concatenate(chunks, axis=2).squeeze()

    def _encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        input_ids = self.tokenizer.encode(text)
        mask = np.ones_like(input_ids)
        hidden = self.t5_encoder.run(
            ["hidden_states"],
            {"input_ids": input_ids.astype(np.int64), "attention_mask": mask.astype(np.int64)},
        )[0]
        return hidden, mask

    def _dit_step(
        self,
        noisy: np.ndarray,
        t: float,
        audio_feat: np.ndarray,
        text_feat: np.ndarray,
        text_mask: np.ndarray,
    ) -> np.ndarray:
        B, T, _ = noisy.shape
        fd = self._fdtype
        inputs = {
            "noisy_audio": noisy.astype(fd),
            "time": np.array([t], dtype=fd),
            "audio_features": audio_feat.astype(fd),
            "text_features": text_feat.astype(fd),
            "text_mask": text_mask.astype(np.bool_),
            "masked_video_features": np.zeros((B, 1024, T), dtype=fd),
            "anchor_ids": np.array([[0, 3]] * B, dtype=np.int64),
            "anchor_alignment": np.zeros((B, T), dtype=np.int64),
            "audio_pad_mask": np.ones((B, T), dtype=np.bool_),
        }
        return self.dit.run(None, inputs)[0]

    def _run_dit_on_chunk(
        self,
        audio_feat_chunk: np.ndarray,
        text_feat: np.ndarray,
        text_mask: np.ndarray,
        chunk_idx: int,
        total_chunks: int,
    ) -> np.ndarray:
        B, T, C = audio_feat_chunk.shape
        x = np.random.randn(B, T, C).astype(np.float32)
        dt = 1.0 / self.num_ode_steps
        for i in range(self.num_ode_steps):
            t = i * dt
            k1 = self._dit_step(x, t, audio_feat_chunk, text_feat, text_mask)
            k2 = self._dit_step(x + k1 * (dt / 2), t + dt / 2, audio_feat_chunk, text_feat, text_mask)
            x = x + k2 * dt
            print(f"    청크 {chunk_idx+1}/{total_chunks} — ODE 스텝 {i+1}/{self.num_ode_steps}", end="\r")
        print()
        return x

    def separate(
        self,
        audio: np.ndarray,
        text: str,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        print("  [a] 오디오 인코딩...")
        latent = self._encode_audio(audio).transpose(0, 2, 1)  # (1, T, 128)
        audio_feat = np.concatenate([latent, latent], axis=2)   # (1, T, 256)

        print("  [b] 텍스트 인코딩...")
        text_feat, text_mask = self._encode_text(text)

        # 25 latent steps per second (48000 / 1920)
        steps_per_sec = 48000 / 1920
        chunk_steps = max(1, int(chunk_seconds * steps_per_sec))
        overlap_steps = max(1, int(overlap_seconds * steps_per_sec))
        T = audio_feat.shape[1]

        if T <= chunk_steps:
            # 전체를 한 번에 처리
            chunks_out = [self._run_dit_on_chunk(audio_feat, text_feat, text_mask, 0, 1)]
            result = chunks_out[0]
        else:
            # 청킹 처리
            stride = chunk_steps - overlap_steps
            starts = list(range(0, T, stride))
            # 마지막 청크가 끝을 포함하도록
            if starts[-1] + chunk_steps < T:
                starts.append(T - chunk_steps)
            n_chunks = len(starts)
            print(f"  [c] ODE 솔버 ({self.num_ode_steps} 스텝, {n_chunks} 청크)...")

            result = np.zeros((1, T, 256), dtype=np.float32)
            weight = np.zeros((T,), dtype=np.float32)

            for ci, start in enumerate(starts):
                end = min(start + chunk_steps, T)
                chunk = audio_feat[:, start:end, :]
                out = self._run_dit_on_chunk(chunk, text_feat, text_mask, ci, n_chunks)

                # crossfade 가중치 (linear ramp at boundaries)
                clen = end - start
                w = np.ones(clen, dtype=np.float32)
                ramp = np.linspace(0, 1, overlap_steps)
                if start > 0:
                    w[:overlap_steps] = ramp
                if end < T:
                    w[-overlap_steps:] = ramp[::-1]

                result[:, start:end, :] += out * w[None, :, None]
                weight[start:end] += w

            result /= weight[None, :, None]

        target_latent = result[:, :, :128].transpose(0, 2, 1)
        residual_latent = result[:, :, 128:].transpose(0, 2, 1)

        print("  [d] 오디오 디코딩...")
        target = self._decode_audio(target_latent)
        residual = self._decode_audio(residual_latent)
        return target, residual


def load_audio(path: Path, target_sr: int = 48000) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze().numpy().astype(np.float32)


def extract_audio(video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "48000", "-ac", "1", str(audio_path)],
        check=True, capture_output=True,
    )


def merge(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
         "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output_path)],
        check=True, capture_output=True,
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    pipeline: SAMAudioONNXPipeline,
    prompt: str,
    target: bool,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
) -> Path:
    suffix = "_target" if target else "_muted"
    output_path = output_dir / f"{video_path.stem}{suffix}{video_path.suffix}"

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        print("  [1/3] 오디오 추출...")
        audio_wav = tmp / "audio.wav"
        extract_audio(video_path, audio_wav)

        print(f'  [2/3] 소리 분리 (프롬프트: "{prompt}")...')
        audio = load_audio(audio_wav)
        target_audio, residual_audio = pipeline.separate(
            audio, prompt, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds
        )

        out_audio = target_audio if target else residual_audio
        separated_wav = tmp / "separated.wav"
        sf.write(str(separated_wav), out_audio, 48000)

        print("  [3/3] 영상 합성...")
        merge(video_path, separated_wav, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="영상에서 특정 소리를 제거합니다 (SAM-Audio ONNX fp16)."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", "-o", type=Path, default=None)
    parser.add_argument(
        "--prompt", "-p", default="person talking near the camera",
        help="제거할 소리 설명",
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=16,
        help="ODE 스텝 수 (기본: 16, 낮을수록 빠름)",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help=f"ONNX 모델 디렉토리 (기본: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--device", "-d", default="cuda", choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=30.0,
        help="DiT 청킹 단위 (초, 기본: 30. 메모리 부족 시 낮춤)",
    )
    parser.add_argument(
        "--overlap-seconds", type=float, default=2.0,
        help="청크 간 crossfade 오버랩 (초, 기본: 2)",
    )
    parser.add_argument(
        "--keep-target", action="store_true",
        help="잔여(residual) 대신 분리된 소리(target)를 출력",
    )
    args = parser.parse_args()

    pipeline = SAMAudioONNXPipeline(
        model_dir=str(args.model_dir),
        device=args.device,
        num_ode_steps=args.steps,
    )

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
                video_path, output_dir, pipeline,
                args.prompt, target=args.keep_target,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
            )
            print(f"  완료 → {result}")
        except Exception as e:
            import traceback
            print(f"  오류: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
