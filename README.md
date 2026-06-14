# AI Video Generation Pipeline

End-to-end automated video production pipeline: script to image generation to neural TTS to AI animation to multi-track audio mixing to final render. Built on a custom deployment of Lightricks' LTX-2.3 (22B parameter) model running on NVIDIA A100-80GB GPUs via Replicate.

## Pipeline Stages

1. **Script Generation** - AI-generated scripts with configurable tone, topic, and character voice
2. **Image Generation** - Scene-by-scene creation with character consistency references
3. **Text-to-Speech** - Neural TTS with customizable voice profiles (Qwen3-TTS)
4. **AI Animation** - Audio-driven lip sync and video generation using LTX-2.3 two-stage distilled pipeline
5. **Audio Mixing** - Multi-track composition: dialogue (100%), SFX (60-80%), music (22%, ducks to 6% during speech)
6. **Final Render** - ffmpeg composition with transitions, H.264 + AAC encode

## The LTX-2.3 Deployment

This pipeline is built around a custom Cog container deployment of LTX-2.3 on Replicate. The model is a 22B parameter distilled video generation model with native audio synthesis.

### Model Architecture

- **Base:** Lightricks LTX-2.3-22B-Distilled-1.1
- **Text Encoder:** Gemma 3 12B (QAT Q4 unquantized, ~23GB VRAM)
- **Video VAE + Transformer:** ~43GB VRAM (BF16) or ~27.5GB (FP8 prequantized)
- **Spatial Upscaler:** 2x resolution upscaler (Stage 2 refinement)
- **LoRA:** Distilled LoRA-384 for quality enhancement
- **Total VRAM:** ~67-70GB on A100-80GB (fits with headroom)

### Two-Stage Pipeline

Stage 1 generates at half resolution (e.g., 384x672 for a 768x1344 target), then Stage 2 upsamples 2x using the spatial upscaler with additional denoising refinement. The distilled model uses only 8 denoising steps for fast inference.

### Working Resolutions

| Resolution | Max Frames | Duration @24fps | Time | Cost |
|-----------|-----------|-----------------|------|------|
| 512x320 | 9 | ~0.4s | 82s | ~$0.02 |
| 768x1344 | 25 | ~1s | 83s | ~$0.02 |
| 1088x1920 | 48 | 2s | 98s | ~$0.02 |

## Critical Bugs Found and Fixed

These were real engineering challenges solved during deployment:

1. **Missing torch.no_grad()** - The #1 memory bug. Without inference mode, every forward pass built a full autograd graph that pinned weights and wasted 17-49GB of VRAM. Adding @torch.inference_mode() and manual torch.no_grad() contexts brought peak usage from ~70GB down to ~45GB.

2. **FP8 Quantization on A100** - A100 GPUs have no FP8 tensor cores. The prequantized FP8 checkpoint (Lightricks/LTX-2.3-fp8) creates double-residency during upcast. Solution: use UPCAST_DURING_INFERENCE flag instead of runtime fp8_cast().

3. **CUDA Version Mismatch** - Cog containers auto-detect CUDA from torch version. Using torch+cu129 caused 8+ failed builds. Switched to torch+cu126 (CUDA 12.6) resolved all build issues.

4. **uv Workspace Conflicts** - The LTX-2 repo uses uv workspaces with cu129 torch. Installing via pip directly (pip install ./packages/ltx-core ./packages/ltx-pipelines) bypasses uv and uses the system torch.

## Audio Pipeline

The audio mixing system (`audio/mix_audio.py`) handles multi-track composition:

- **Dialogue:** Always 100% volume (loudest element)
- **SFX:** 60-80% volume, placed at specific timestamps (e.g., sparkle_shimmer at 0.5s, chainsaw_rev at 62.0s)
- **Background Music:** 22% during silence, automatically ducks to 6% during speech (sidechain compression)
- **Output:** 44.1kHz stereo WAV, mixed via numpy with ffmpeg encoding

SFX library includes 22 synthesized effects: impacts, transitions, foley, weapons, stingers, UI sounds, and vehicle sounds.

## Tech Stack

- **Model:** LTX-2.3-22B-Distilled-1.1 (Lightricks)
- **GPU:** NVIDIA A100-80GB (Replicate cloud GPUs)
- **Container:** Cog (Docker + CUDA 12.6)
- **Framework:** PyTorch 2.7.1
- **Text Encoder:** Gemma 3 12B QAT
- **TTS:** Qwen3-TTS with custom voice presets
- **Video:** ffmpeg (H.264/AAC encode)
- **Audio:** numpy + ffmpeg (multi-track mixing with sidechain ducking)
- **CI/CD:** GitHub Actions to Replicate registry
- **Build System:** PlatformIO

## Project Structure

```
ai-video-generation-pipeline/
  src/
    predict.py        # Cog predictor - LTX-2.3 inference on A100
    pipeline.py       # End-to-end orchestration (gen + lip sync + audio + render)
    cog.yaml          # Docker config (CUDA 12.6, Python 3.12, system deps)
    requirements.txt  # torch 2.7.1+cu126, cog, Pillow
  audio/
    mix_audio.py      # Multi-track audio mixer with sidechain ducking
  docs/
    LTX_README.md     # LTX-2.3 model documentation
    RESEARCH.md       # Full deployment research report (73GB weights, build history, architecture)
```

## Performance

- **Cost per generation:** ~$0.02 (Replicate A100 pricing)
- **Cold start:** ~10 min (downloads 73GB of weights: 51GB LTX + 23GB Gemma)
- **Warm inference:** 82-98s per video depending on resolution
- **VRAM:** ~45-70GB peak depending on FP8 vs BF16 mode
- **Docker image:** ~6GB (weights download at runtime, not baked in)

## License

MIT

---

Built by [Molt Studios](https://github.com/moltstudios)
