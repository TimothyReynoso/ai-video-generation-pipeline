# LTX-2.3 Cog Model for Replicate

Custom deployment of Lightricks' LTX-2.3 (22B) audio-video generation model.

## Features
- **Text-to-Video with Audio**: Generate video + synchronized audio from text
- **Image-to-Video with Audio**: Animate a still image with generated audio
- **Two-stage pipeline**: Stage 1 generates, Stage 2 upscales 2x
- **Distilled model**: 8 denoising steps for fast inference
- **FP8 quantization**: Fits on single A100 (80GB)

## Model Details
- **Base**: Lightricks LTX-2.3-22B-Distilled-1.1
- **Parameters**: 22 Billion
- **License**: Free for under $10M revenue
- **Native audio**: Yes (synchronized speech, music, sound effects)
