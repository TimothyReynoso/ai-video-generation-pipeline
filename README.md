# AI Video Generation Pipeline

End-to-end automated video production: script to image generation to TTS to AI animation to audio mixing to final video. Custom model containers deployed on cloud GPUs.

## Pipeline Stages
1. Script Generation - AI-generated scripts with configurable tone and topic
2. Image Generation - Scene-by-scene creation with character consistency
3. Text-to-Speech - Neural TTS with customizable voice profiles
4. AI Animation - Audio-driven lip sync using LTX-2.3
5. Audio Mixing - Multi-track: dialogue (100%), SFX (60-80%), music (22%)
6. Final Render - ffmpeg composition with transitions

## Key Technical Work
- LTX-2.3 deployed on Replicate (Cog containers) with A100-80GB optimization
- Found and fixed critical bug: Missing torch.no_grad() was building autograd graphs, pinning 17-49GB of unnecessary memory
- FP8 quantization fix: A100 has no FP8 tensor cores - switched to upcast during inference
- Cost: ~$0.02 per scene generation

## Tech Stack
- Model: LTX-2.3 (Lightricks)
- GPU: NVIDIA A100-80GB (Replicate, Modal)
- Container: Cog (Docker + CUDA 12.6)
- Framework: PyTorch 2.7.1
- Video: ffmpeg
- TTS: Qwen3-TTS with custom presets

## License
MIT
