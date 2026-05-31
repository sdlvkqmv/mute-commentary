import argparse
import subprocess
import sys
from pathlib import Path

from mlx_audio.sts import SAMAudio, SAMAudioProcessor, save_audio
import mlx.core as mx


def extract_audio(video_path: Path) -> Path:
    mp3_path = video_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "48000", "-ac", "1", "-q:a", "0", str(mp3_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp3_path


parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input video file")
parser.add_argument("--description", default="woman speaking", help="Audio description to isolate")
args = parser.parse_args()

input_path = Path(args.input)
if not input_path.exists():
    print(f"Error: {input_path} not found")
    sys.exit(1)

print(f"Extracting audio from {input_path}...")
mp3_path = extract_audio(input_path)
print(f"Saved to {mp3_path}")

# Load model and processor
processor = SAMAudioProcessor.from_pretrained("mlx-community/sam-audio-large-fp16")
model = SAMAudio.from_pretrained("mlx-community/sam-audio-large-fp16")

# Process inputs
batch = processor(
    descriptions=[args.description],
    audios=[str(mp3_path)],
    # anchors=[[("+", 0.2, 0.5)]],  # Optional: temporal
)

# Separate audio
result = model.separate(
    audios=batch.audios,
    descriptions=batch.descriptions,
    sizes=batch.sizes,
    anchor_ids=batch.anchor_ids,
    anchor_alignment=batch.anchor_alignment,
    ode_decode_chunk_size=50,
)

# For long audio files, use separate_long().
# Note: This is slower than separate() but it is more memory efficient.
#result = model.separate_long(
#     audios=batch.audios,
#     descriptions=batch.descriptions,
#     chunk_seconds=10.0,
#     overlap_seconds=3.0,
##     anchor_ids=batch.anchor_ids,
 #    anchor_alignment=batch.anchor_alignment,
 #    ode_decode_chunk_size=50,
#)

stem = input_path.stem
save_audio(result.target[0], f"{stem}_separated.wav", sample_rate=model.sample_rate)
save_audio(result.residual[0], f"{stem}_residual.wav", sample_rate=model.sample_rate)

print("Merging residual audio with original video...")
output_video = f"{stem}_muted.mp4"
subprocess.run(
    [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-i", f"{stem}_residual.wav",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-shortest",
        output_video,
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

print(f"Peak memory: {result.peak_memory:.2f} GB")
print(f"Output: {stem}_separated.wav, {stem}_residual.wav, {output_video}")
