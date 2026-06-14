#!/usr/bin/env python3
"""
Audio mixing script for video post-production.
Layers SFX + background music onto rendered video with sidechain ducking.

Usage:
  python3 mix_audio.py --video input.mp4 --music bg_beat.wav \
    --sfx "sparkle_shimmer:0.5:0.8" "chainsaw_rev:62.0:0.9" \
    --output final_remixed.mp4

SFX format: "sfx_name:delay_seconds:volume"
  sfx_name: filename without extension from ~/agency/shared/sfx-library/
  delay_seconds: when the SFX plays (from video start)
  volume: 0.0-1.0

Audio mixing rules:
  - Dialogue: 100% (always loudest)
  - SFX: configured per-placement (typically 60-90%)
  - Music: 22% during silence, ducks to 6% during dialogue
"""

import argparse
import subprocess
import struct
import os
import sys
import numpy as np

SR = 44100
SFX_DIR = os.path.dirname(os.path.abspath(__file__))

def load_audio(path):
    """Load any audio file to numpy float32 stereo array."""
    result = subprocess.run(
        ['ffmpeg', '-i', path, '-ar', str(SR), '-ac', '2', '-f', 'wav', 'pipe:1'],
        capture_output=True, timeout=60
    )
    if result.returncode != 0:
        print(f"  ⚠️ Could not load: {path}")
        return np.zeros((SR, 2), dtype=np.float32)
    data = result.stdout
    idx = data.find(b'data')
    if idx < 0:
        return np.zeros((SR, 2), dtype=np.float32)
    header_size = idx + 8
    raw = data[header_size:]
    if len(raw) < 4:
        return np.zeros((SR, 2), dtype=np.float32)
    pcm = np.frombuffer(raw, dtype=np.int16)
    pcm = pcm.reshape(-1, 2)
    return pcm.astype(np.float32) / 32768.0

def save_wav(data, path):
    """Save numpy float32 stereo array as WAV."""
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    n_frames = len(pcm)
    byte_rate = SR * 4
    data_size = n_frames * 4
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 2, SR, byte_rate, 4, 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(pcm.tobytes())

def mix_at(base, sfx, delay_s, volume=0.7):
    """Mix an SFX into base array at given delay."""
    delay_samples = int(delay_s * SR)
    end = min(delay_samples + len(sfx), len(base))
    length = end - delay_samples
    if length > 0 and delay_samples < len(base):
        base[delay_samples:end] += sfx[:length] * volume

def find_sfx(name):
    """Find SFX file by name in the library."""
    for root, dirs, files in os.walk(SFX_DIR):
        for f in files:
            if f.startswith(name) and f.endswith('.wav'):
                return os.path.join(root, f)
    # Try with .mp3
    for root, dirs, files in os.walk(SFX_DIR):
        for f in files:
            if f.startswith(name) and f.endswith('.mp3'):
                return os.path.join(root, f)
    return None

def main():
    parser = argparse.ArgumentParser(description='Mix SFX + music into video')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--music', default=None, help='Background music file')
    parser.add_argument('--sfx', nargs='*', default=[], help='SFX placements: "name:delay:volume"')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--music-vol', type=float, default=0.22, help='Music volume (0-1)')
    parser.add_argument('--music-duck', type=float, default=0.06, help='Music duck volume during speech')
    parser.add_argument('--dialogue-threshold', type=float, default=0.03, help='Speech detection threshold')
    parser.add_argument('--crf', type=int, default=23, help='Video compression quality (lower=better)')
    args = parser.parse_args()

    # Get video duration
    dur_result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', args.video],
        capture_output=True, text=True
    )
    duration = float(dur_result.stdout.strip())
    target_frames = int(duration * SR)
    print(f"🎬 Video: {duration:.1f}s ({target_frames} samples)")

    # Load dialogue from video
    print("📥 Loading dialogue...")
    tmp_wav = '/tmp/_mix_dialogue.wav'
    subprocess.run(['ffmpeg', '-y', '-i', args.video, '-vn', '-ar', str(SR), '-ac', '2', tmp_wav],
                   capture_output=True, timeout=30)
    dialogue = load_audio(tmp_wav)
    
    # Pad/trim to match
    if len(dialogue) < target_frames:
        padded = np.zeros((target_frames, 2), dtype=np.float32)
        padded[:len(dialogue)] = dialogue
        dialogue = padded
    else:
        dialogue = dialogue[:target_frames]
    print(f"  Dialogue: {len(dialogue)/SR:.1f}s")

    # Create SFX track
    print("📥 Layering SFX...")
    sfx_track = np.zeros((target_frames, 2), dtype=np.float32)
    for sfx_spec in args.sfx:
        parts = sfx_spec.split(':')
        if len(parts) != 3:
            print(f"  ⚠️ Bad SFX spec: {sfx_spec} (need name:delay:volume)")
            continue
        name, delay_s, volume = parts[0], float(parts[1]), float(parts[2])
        sfx_path = find_sfx(name)
        if sfx_path:
            sfx = load_audio(sfx_path)
            mix_at(sfx_track, sfx, delay_s, volume)
            print(f"  ✅ {name} at {delay_s:.1f}s vol={volume}")
        else:
            print(f"  ❌ SFX not found: {name}")

    # Load music
    music_padded = np.zeros((target_frames, 2), dtype=np.float32)
    if args.music:
        print("📥 Loading music...")
        music = load_audio(args.music)
        music_padded[:min(len(music), target_frames)] = music[:target_frames]

    # Final mix with sidechain ducking
    print("🎛️ Mixing with ducking...")
    final = np.zeros((target_frames, 2), dtype=np.float32)
    final += dialogue * 1.0
    final += sfx_track * 0.8

    # Duck music during dialogue
    window = int(SR * 0.1)  # 100ms windows
    for i in range(0, target_frames, window):
        end = min(i + window, target_frames)
        dialog_energy = np.max(np.abs(dialogue[i:end])) if i < len(dialogue) else 0
        vol = args.music_duck if dialog_energy > args.dialogue_threshold else args.music_vol
        final[i:end] += music_padded[i:end] * vol

    # Normalize to prevent clipping
    mx = np.max(np.abs(final))
    if mx > 0.95:
        final = final * 0.95 / mx

    # Save mixed audio
    print("💾 Saving mixed audio...")
    mix_path = '/tmp/_mix_final.wav'
    save_wav(final, mix_path)

    # Mux with video
    print("🎬 Muxing with video...")
    cmd = [
        'ffmpeg', '-y',
        '-i', args.video,
        '-i', mix_path,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', str(args.crf),
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v', '-map', '1:a',
        '-movflags', '+faststart',
        args.output
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"❌ ffmpeg failed: {result.stderr.decode()[-500:]}")
        sys.exit(1)

    size = os.path.getsize(args.output)
    print(f"\n✅ Done! {args.output} ({size//1048576}MB, {duration:.1f}s)")

    # Cleanup
    for tmp in [tmp_wav, mix_path]:
        if os.path.exists(tmp):
            os.unlink(tmp)

if __name__ == '__main__':
    main()
