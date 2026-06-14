# AI Video Generation Pipeline (LTX-2.3)

Custom deployment of Lightricks' LTX-2.3 (22B parameter) video generation model on NVIDIA A100-80GB GPUs via Replicate. Features FP8 quantization, two-stage pipeline with 2x upscaling, and distilled 8-step inference. This was the foundational ML engineering work that preceded our production avatar pipeline.

> **Note:** This repo covers the custom model deployment and ML engineering. The production avatar video system built on top of this research lives in [avatar-video-pipeline](https://github.com/TimothyReynoso/avatar-video-pipeline).

## Model Details

- **Base:** Lightricks LTX-2.3-22B-Distilled-1.1
- **Parameters:** 22 Billion
- **Quantization:** FP8 (prequantized checkpoint, 27.5GB vs 43GB bf16)
- **License:** Free for commercial use
- **Deployment:** Custom Cog container on Replicate

## Pipeline Stages

1. **Script Generation** - AI-generated scripts with configurable tone, topic, and character voice
2. **Image Generation** - Scene-by-scene creation with character consistency references
3. **Text-to-Speech** - Neural TTS with customizable voice profiles (Qwen3-TTS)
4. **AI Animation** - Audio-driven lip sync using LTX-2.3 A2VidPipelineTwoStage
5. **Audio Mixing** - Multi-track: dialogue (100%), SFX (60-80%), music (22%)
6. **Final Render** - ffmpeg concat, re-mux, 9:16 portrait export

## The Three Big Bugs (Found and Fixed)

These were real production bugs that took days to diagnose. Documenting them here because they represent deep PyTorch/CUDA debugging work.

### Bug 1: Missing `torch.no_grad()`
The #1 memory bug. Without `torch.no_grad()`, every forward pass built a full autograd graph that pinned weights and wasted 17-49GB of VRAM. The LTX pipeline's `@torch.inference_mode()` was only applied to the CLI `main()` function, not the library API.

### Bug 2: FP8 Double-Residency
The prequantized FP8 checkpoint (`Lightricks/LTX-2.3-fp8`) uses `UPCAST_DURING_INFERENCE` mode. A100 GPUs don't have FP8 tensor cores, so `fp8_cast()` creates double-residency (both fp8 and bf16 copies in VRAM simultaneously). Solution: use the prequantized checkpoint with the correct inference flag rather than runtime casting.

### Bug 3: Missing `@torch.inference_mode()`
When using the pipeline as a library (not via CLI), the decorator was missing. Added explicit `with torch.inference_mode():` around the generation call.

## Performance

| Resolution | Max Frames | Duration @8fps | Time | Cost |
|-----------|-----------|----------------|------|------|
| 512x320 | 9 | ~1s | 82s | ~$0.02 |
| 768x1344 | 25 | ~3s | 83s | ~$0.02 |
| 1088x1920 | 48 | 6s | 98s | ~$0.02 |

## Hard Limits

- **VAE decode ceiling:** ~48 frames at 1088x1920, ~25 at 768x1344
- **Resolution:** Must be divisible by 64
- **Tile overlap:** Must be divisible by 32
- **Temporal tile size:** Must be divisible by 8
- **Container reuse:** OOM'd containers leak GPU memory; wait or push new build

## Audio Architecture

Multi-track mixing with precise volume control:
- Dialogue: 100% (always loudest)
- SFX: 60-80% (punchy, doesn't overpower speech)
- Music: 22% (ducks to ~6% during speech via sidechain compression)

## Source Code

| File | Lines | Purpose |
|------|-------|---------|
| `predict.py` | 771 | Main Cog predictor with all monkey-patches and memory fixes |
| `pipeline.py` | 249 | End-to-end orchestration (script to final video) |
| `mix_audio.py` | 202 | Multi-track audio mixer |
| `docs/RESEARCH.md` | 600+ | Full deployment research report |
| `docs/LTX_README.md` | - | Model documentation and usage |
| `cog.yaml` | - | Docker build config (CUDA 12.6, torch 2.7.1) |

## Tech Stack

- **Model:** LTX-2.3-22B-Distilled-1.1 (Lightricks)
- **Container:** Cog (Replicate's ML deployment framework)
- **GPU:** NVIDIA A100-80GB (via Replicate)
- **Framework:** PyTorch 2.7.1 + CUDA 12.6
- **Quantization:** FP8 prequantized checkpoint
- **Pipeline:** A2VidPipelineTwoStage (audio-to-video with 2x upscale)
- **Audio:** Multi-track ffmpeg pipeline

## What This Repo Demonstrates

- Deep PyTorch/CUDA memory debugging (VRAM profiling, autograd graph analysis)
- Custom ML model deployment on cloud GPUs (Cog, Replicate, Docker)
- Quantization strategy (FP8 vs bf16 tradeoffs, double-residency bugs)
- End-to-end ML pipeline design (script to final video output)
- Production audio engineering (multi-track mixing, sidechain compression)

## License

MIT

---

Built by [Molt Studios](https://github.com/moltstudios)
