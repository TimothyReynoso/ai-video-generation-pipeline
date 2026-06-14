"""
LTX-2.3 Cog Predictor — Prequantized FP8 Build
================================================
Uses Lightricks' official prequantized FP8 checkpoint (27.5GB vs 43GB bf16).
The FP8 checkpoint has pre-computed scale keys baked in, so weights are never
materialized as bf16 on the GPU — eliminates the double-residency bug that
caused 70.4GB usage with runtime fp8-cast.

Memory budget (single A100 80GB):
  Transformer (fp8 + LoRA): ~35GB
  Gemma (bf16, freed before transformer): ~22GB
  Activations + context: ~10GB
  Peak: ~45GB — fits with 35GB headroom

All downloads in predict() for "Processing" status + visible logs.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import gc
import math
from contextlib import contextmanager
import subprocess
import random
import time
from typing import Optional

import torch
from cog import BasePredictor, Input, Path, Secret

MODEL_DIR = "/opt/models"

# PREQUANTIZED FP8 checkpoint — 27.5GB vs 43GB bf16
FP8_CHECKPOINT = f"{MODEL_DIR}/ltx-2.3-22b-distilled-fp8.safetensors"
# BF16 checkpoint as fallback (NOT used by default)
BF16_CHECKPOINT = f"{MODEL_DIR}/ltx-2.3-22b-distilled-1.1.safetensors"
DISTILLED_LORA = f"{MODEL_DIR}/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
LIPDUB_LORA = f"{MODEL_DIR}/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
SPATIAL_UPSCALER = f"{MODEL_DIR}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = f"{MODEL_DIR}/gemma"


def _file_exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def _gpu_mem(label=""):
    """Log current GPU memory state."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU {label}: {alloc:.1f}GB alloc, {reserved:.1f}GB reserved", flush=True)


def _download_ltx(token=None):
    """Download LTX-2.3 weights via pget (public, no auth)."""
    # Download FP8 checkpoint from the fp8 repo
    fp8_files = {
        "ltx-2.3-22b-distilled-fp8.safetensors": (FP8_CHECKPOINT, "Lightricks/LTX-2.3-fp8"),
    }
    for filename, (dest, repo) in fp8_files.items():
        if _file_exists(dest):
            sz = os.path.getsize(dest) / (1024**3)
            print(f"  ✓ {filename} ({sz:.1f}GB)", flush=True)
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        print(f"  ↓ {filename} via pget...", flush=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        subprocess.check_call(["pget", "-f", url, dest], timeout=1800)
        sz = os.path.getsize(dest) / (1024**3)
        print(f"  ✓ {filename} ({sz:.1f}GB)", flush=True)

    # Download LoRA and upscaler from the main repo
    other_files = {
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors": (DISTILLED_LORA, "Lightricks/LTX-2.3"),
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": (SPATIAL_UPSCALER, "Lightricks/LTX-2.3"),
    }
    # LipDub IC-LoRA from separate repo
    lipdub_files = {
        "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors": (LIPDUB_LORA, "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub"),
    }
    for filename, (dest, repo) in other_files.items():
        if _file_exists(dest):
            sz = os.path.getsize(dest) / (1024**3)
            print(f"  ✓ {filename} ({sz:.1f}GB)", flush=True)
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        print(f"  ↓ {filename} via pget...", flush=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        subprocess.check_call(["pget", "-f", url, dest], timeout=1800)
        sz = os.path.getsize(dest) / (1024**3)
        print(f"  ✓ {filename} ({sz:.1f}GB)", flush=True)

    # LipDub LoRA is in a gated repo — needs huggingface_hub with token
    for filename, (dest, repo) in lipdub_files.items():
        if _file_exists(dest):
            sz = os.path.getsize(dest) / (1024**3)
            print(f"  ✓ {filename} ({sz:.1f}GB)", flush=True)
            continue
        print(f"  ↓ {filename} via huggingface_hub (gated)...", flush=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo, filename=filename, local_dir=os.path.dirname(dest), token=token)
        # hf_hub_download saves with the repo structure, move to our path
        downloaded = os.path.join(os.path.dirname(dest), filename)
        if downloaded != dest and os.path.exists(downloaded):
            os.rename(downloaded, dest)
        sz = os.path.getsize(dest) / (1024**3)
        print(f"  ✓ {filename} ({sz:.1f}GB)", flush=True)


def _download_gemma(token):
    """Download Gemma 12B via huggingface_hub (gated, needs auth)."""
    from huggingface_hub import snapshot_download
    print(f"  ↓ Gemma 12B via huggingface_hub...", flush=True)
    snapshot_download(
        repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
        local_dir=GEMMA_ROOT,
        token=token,
        ignore_patterns=["*.msgpack", "*.bin", "*.h5", "*.ot", "onnx/*"],
    )
    print(f"  ✓ Gemma 12B downloaded", flush=True)


class Predictor(BasePredictor):
    def setup(self):
        self.pipeline = None
        self.tiling_config = None
        print("SETUP: done", flush=True)

    def _interpolate_frames(self, input_path: str, target_fps: int) -> str:
        """Interpolate video frames to target FPS using RIFE (GPU) or ffmpeg fallback."""
        import cv2

        cap = cv2.VideoCapture(input_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"  Source: {total_frames}f @ {source_fps:.0f}fps → Target: {target_fps}fps", flush=True)

        # Try RIFE first (GPU-accelerated, ~20s)
        interp_path = None
        rife_weights = os.path.join(os.path.dirname(__file__), "rife", "flownet.pkl")
        if os.path.isfile(rife_weights):
            try:
                interp_path = self._rife_interpolate(input_path, source_fps, target_fps, w, h, total_frames)
                print(f"  RIFE interpolation succeeded", flush=True)
            except Exception as e:
                print(f"  RIFE failed: {e}, falling back to ffmpeg", flush=True)
                interp_path = None

        # Fallback: ffmpeg minterpolate (CPU-only, ~160s)
        if interp_path is None:
            print(f"  Using ffmpeg minterpolate (slower)", flush=True)
            interp_path = "/tmp/interp_raw.mp4"
            r = subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-an",
                interp_path
            ], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  ffmpeg error: {r.stderr[:300]}", flush=True)
                raise RuntimeError("ffmpeg interpolation failed")

        # Merge original audio back into interpolated video
        output_path = "/tmp/rife_output.mp4"
        r2 = subprocess.run([
            "ffmpeg", "-y",
            "-i", interp_path,
            "-i", input_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_path
        ], capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"  ffmpeg merge warning: {r2.stderr[:300]}", flush=True)
            output_path = interp_path

        # Verify output
        cap2 = cv2.VideoCapture(output_path)
        out_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = cap2.get(cv2.CAP_PROP_FPS)
        cap2.release()
        print(f"  Output: {out_frames}f @ {out_fps:.0f}fps", flush=True)

        return output_path

    def _rife_interpolate(self, input_path: str, source_fps: float, target_fps: int, w: int, h: int, total_frames: int) -> str:
        """GPU-accelerated frame interpolation using vendored RIFE v4.26."""
        import sys
        import cv2
        import math

        rife_dir = os.path.join(os.path.dirname(__file__), "rife")
        device = torch.device('cuda')

        # Calculate interpolation multiplier (power of 2)
        ratio = target_fps / source_fps
        exp = max(1, round(math.log2(ratio)))
        actual_fps = source_fps * (2 ** exp)
        print(f"  RIFE: 2^{exp}x interpolation → {actual_fps:.0f}fps", flush=True)

        # Load RIFE model from vendored package
        from rife import Model as RIFEModel
        model = RIFEModel()
        model.load_model(rife_dir, -1)
        model.eval()
        model.device()
        print(f"  RIFE model loaded", flush=True)

        # Read all frames as tensors
        cap = cv2.VideoCapture(input_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Normalize to [0,1] float tensor on GPU
            tensor = (torch.tensor(frame.transpose(2, 0, 1)).to(device).float() / 255.0).unsqueeze(0)
            frames.append(tensor)
        cap.release()
        print(f"  Read {len(frames)} source frames", flush=True)

        # Multi-pass RIFE interpolation
        # Each pass doubles frame count by inserting intermediate frames
        with torch.no_grad():
            for pass_num in range(exp):
                new_frames = []
                for i in range(len(frames) - 1):
                    new_frames.append(frames[i])
                    mid = model.inference(frames[i], frames[i + 1], timestep=0.5)
                    new_frames.append(mid)
                new_frames.append(frames[-1])
                frames = new_frames
                print(f"  Pass {pass_num + 1}/{exp}: {len(frames)} frames", flush=True)

        print(f"  Final: {len(frames)} frames @ {actual_fps:.0f}fps", flush=True)

        # Write frames as PNG sequence, then encode with ffmpeg for H.264 compatibility
        # cv2.VideoWriter mp4v codec creates green-screen videos in browsers
        frames_dir = "/tmp/rife_frames"
        os.makedirs(frames_dir, exist_ok=True)
        for idx, frame_tensor in enumerate(frames):
            frame_np = (frame_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            cv2.imwrite(f"{frames_dir}/{idx:06d}.png", frame_np)
        print(f"  Wrote {len(frames)} frames to disk", flush=True)

        output_path = "/tmp/interp_raw.mp4"
        r = subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(actual_fps),
            "-i", f"{frames_dir}/%06d.png",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ffmpeg encode error: {r.stderr[:300]}", flush=True)
            raise RuntimeError("ffmpeg encode failed")

        # Cleanup
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)

        # Cleanup GPU memory
        del model, frames
        gc.collect()
        torch.cuda.empty_cache()

        return output_path

    def predict(
        self,
        prompt: str = Input(
            description='Text prompt for video generation.',
            default="A cat sitting on a windowsill watching rain fall outside.",
        ),
        hf_token: Secret = Input(
            description="HuggingFace token with Gemma access",
        ),
        image: Optional[Path] = Input(description="Image for img2vid.", default=None),
        audio: Optional[Path] = Input(
            description="Audio file for lipsync. When provided, generates video with lip movements synced to this audio.",
            default=None,
        ),
        image_strength: float = Input(description="Image strength (0-1).", default=1.0),
        width: int = Input(description="Width (must be divisible by 32).", default=768),
        height: int = Input(description="Height (must be divisible by 32).", default=1344),
        num_frames: int = Input(description="Frames (must be 8k+1). 97≈4s, 41≈1.7s.", default=97),
        frame_rate: float = Input(description="FPS.", default=24.0),
        seed: int = Input(description="Seed (-1=random).", default=-1),
        enhance_prompt: bool = Input(description="Enhance prompt.", default=False),
        target_fps: int = Input(
            description="Target FPS after RIFE frame interpolation (0=disabled). Set to 45 or 60 for smooth output.",
            default=0,
        ),
        lip_sync_strength: float = Input(
            description="LatentSync guidance scale. 1.0=conservative (fewer artifacts), 1.5=stronger lip sync, 2.0=aggressive.",
            default=1.5,
        ),
    ) -> Path:
        """Generate video with optional audio-driven lipsync using prequantized FP8 checkpoint."""
        token = hf_token.get_secret_value() if hasattr(hf_token, 'get_secret_value') else str(hf_token)

        # --- Step 1: Download weights ---
        if not _file_exists(FP8_CHECKPOINT) or not _file_exists(DISTILLED_LORA):
            print("=== DOWNLOADING LTX-2.3 FP8 (~35GB) ===", flush=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            start = time.time()
            _download_ltx(token=token)
            print(f"=== LTX downloaded in {time.time()-start:.0f}s ===", flush=True)
        else:
            print("=== LTX FP8 already cached ===", flush=True)

        if not _file_exists(os.path.join(GEMMA_ROOT, "model-00001-of-00005.safetensors")):
            print("=== DOWNLOADING Gemma 12B (~22GB, gated) ===", flush=True)
            os.makedirs(GEMMA_ROOT, exist_ok=True)
            start = time.time()
            _download_gemma(token)
            print(f"=== Gemma downloaded in {time.time()-start:.0f}s ===", flush=True)
        else:
            print("=== Gemma already cached ===", flush=True)

        # --- Step 2: Load pipeline ---
        # Always loads DistilledPipeline for Pass 1 (base video generation)
        # LipDubPipeline is loaded inline in Pass 2 when audio is provided

        if self.pipeline is None:
            from ltx_core.loader import LoraPathStrengthAndSDOps
            from ltx_core.model.video_vae import TilingConfig
            from ltx_pipelines.distilled import DistilledPipeline
            from ltx_core.quantization.fp8_cast import build_policy as fp8_build_policy

            start = time.time()

            loras = (
                LoraPathStrengthAndSDOps(
                    path=DISTILLED_LORA, strength=0.8, sd_ops=None,
                ),
            )

            print(f"  Checkpoint: {FP8_CHECKPOINT}", flush=True)
            print(f"  Checkpoint size: {os.path.getsize(FP8_CHECKPOINT)/(1024**3):.1f}GB", flush=True)
            print("  Building FP8 policy from prequantized checkpoint...", flush=True)
            quantization = fp8_build_policy(FP8_CHECKPOINT)
            print(f"  Policy: sd_ops={quantization.sd_ops.name}, mappings={len(quantization.sd_ops.mapping)}", flush=True)

            # Always load DistilledPipeline first (used for Pass 1 base generation)
            # LipDubPipeline is loaded inline in Pass 2 when audio is provided
            self.pipeline = DistilledPipeline(
                distilled_checkpoint_path=FP8_CHECKPOINT,
                gemma_root=GEMMA_ROOT,
                spatial_upsampler_path=SPATIAL_UPSCALER,
                loras=loras,
                quantization=quantization,
            )

            self.tiling_config = TilingConfig.default()
            _gpu_mem("after pipeline init")
            print(f"=== DistilledPipeline ready in {time.time()-start:.0f}s ===", flush=True)

        # --- Step 3: Monkey-patch for phase-boundary memory logging ---
        import ltx_pipelines.utils.blocks as _blocks
        _orig_build = _blocks.DiffusionStage._build_transformer
        def _build_with_log(self_stage, **kwargs):
            _gpu_mem("BEFORE transformer build")
            model = _orig_build(self_stage, **kwargs)
            # Log param dtypes summary
            fp8_count = sum(1 for _, p in model.named_parameters() if 'float8' in str(p.dtype))
            bf16_count = sum(1 for _, p in model.named_parameters() if p.dtype == torch.bfloat16)
            total = sum(1 for _ in model.named_parameters())
            total_bytes = sum(p.numel() * p.element_size() for _, p in model.named_parameters())
            print(f"  TRANSFORMER: {total} params, {fp8_count} fp8, {bf16_count} bf16, {total_bytes/1e9:.1f}GB", flush=True)
            _gpu_mem("AFTER transformer build")
            return model
        _blocks.DiffusionStage._build_transformer = _build_with_log

        # =================================================================
        # FIX: Missing torch.inference_mode() in Gemma encode.
        #
        # ROOT CAUSE (Opus diagnosis, confirmed by logs):
        #   - encode() runs WITHOUT inference_mode/no_grad
        #   - Gemma forward builds full autograd graph → 17.4GB extra
        #   - Output hidden states carry grad_fn → pin Gemma's weights
        #   - del text_encoder frees nothing because graph holds weights
        #   - Moving hidden states to CPU caused device mismatch crash
        #
        # FIX:
        #   1. Wrap encode in torch.inference_mode() → kills autograd graph
        #   2. Keep hidden states on GPU (detach only) → ~15MB, not worth moving
        #   3. del text_encoder now works because no graph pins the weights
        #
        # Expected: encode delta ~MB not 17GB, post-free ~0-1GB not 41GB
        # =================================================================
        _orig_pe_call = _blocks.PromptEncoder.__call__

        def _fixed_encode(self_enc, prompts, **kwargs):
            from ltx_pipelines.utils.helpers import cleanup_memory, generate_enhanced_prompt
            from ltx_pipelines.utils.gpu_model import gpu_model
            import logging
            logger = logging.getLogger(__name__)

            # Build and run Gemma text encoder
            logger.info("Building text encoder from %s", self_enc._gemma_root)
            _gpu_mem("BEFORE Gemma build")
            text_encoder = self_enc._build_text_encoder()
            first_dtype = next(text_encoder.parameters()).dtype
            first_dev = next(text_encoder.parameters()).device
            print(f"  Gemma dtype: {first_dtype}, device: {first_dev}", flush=True)
            _gpu_mem("AFTER Gemma build")

            try:
                if kwargs.get('enhance_first_prompt'):
                    prompts = list(prompts)
                    prompts[0] = generate_enhanced_prompt(
                        text_encoder, prompts[0],
                        kwargs.get('enhance_prompt_image'),
                        seed=kwargs.get('enhance_prompt_seed', 42)
                    )
                # THE FIX: inference_mode prevents autograd graph
                with torch.inference_mode():
                    raw_outputs = [text_encoder.encode(p) for p in prompts]
                _gpu_mem("AFTER Gemma encode (inference_mode)")
            finally:
                # Now free works — no autograd graph pinning weights
                text_encoder.to("meta")
                del text_encoder
                gc.collect()
                torch.cuda.empty_cache()
                _gpu_mem("AFTER Gemma force-freed")

            # Detach hidden states but keep on GPU — they're ~15MB,
            # moving to CPU caused device mismatch crash for no benefit
            cleaned_outputs = []
            for hs, mask in raw_outputs:
                cleaned_hs = tuple(h.detach() for h in hs) if isinstance(hs, tuple) else hs.detach()
                cleaned_outputs.append((cleaned_hs, mask.detach()))
            del raw_outputs
            gc.collect()
            _gpu_mem("AFTER hidden states detached (on GPU)")

            # Build embeddings processor
            logger.info("Text encoder done, building embeddings processor")
            _gpu_mem("BEFORE embeddings processor")
            with gpu_model(self_enc._build_embeddings_processor()) as embeddings_processor:
                result = [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in cleaned_outputs]
            _gpu_mem("AFTER embeddings processor freed")

            del cleaned_outputs
            cleanup_memory()
            _gpu_mem("AFTER final cleanup")
            return result

        _blocks.PromptEncoder.__call__ = _fixed_encode

        # Also patch gpu_model to log on every free
        import ltx_pipelines.utils.gpu_model as _gm
        _orig_gm = _gm.gpu_model
        @contextmanager
        def _gm_with_log(model):
            model_name = type(model).__name__
            _gpu_mem(f"BEFORE {model_name}")
            with _orig_gm(model) as m:
                yield m
            _gpu_mem(f"AFTER {model_name} freed")
        _gm.gpu_model = _gm_with_log

        # --- Step 4: Generate ---
        use_audio = audio is not None
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        images = []
        if image is not None:
            from ltx_pipelines.utils.args import ImageConditioningInput
            images = [ImageConditioningInput(path=str(image), frame_idx=0, strength=image_strength)]

        from ltx_core.model.video_vae import get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        print(f"=== GENERATING: {width}x{height}, {num_frames}f, seed={seed}, mode={'LIPDUB+LATENTSYNC' if use_audio else 'I2V'} ===", flush=True)
        _gpu_mem("before generate")
        start = time.time()

        # CRITICAL: Wrap in inference_mode to prevent autograd graph buildup
        with torch.no_grad():
            # ============================================================
            # PASS 1: Generate base video with DistilledPipeline
            # This preserves the character's appearance from the reference image
            # ============================================================
            print(f"=== PASS 1: Base video generation ===", flush=True)
            video, audio_out = self.pipeline(
                prompt=prompt, seed=seed, height=height, width=width,
                num_frames=num_frames, frame_rate=frame_rate,
                images=images, tiling_config=self.tiling_config,
                enhance_prompt=enhance_prompt,
            )

            # Encode base video (no audio yet)
            base_path = "/tmp/base_output.mp4"
            chunks = get_video_chunks_number(num_frames, self.tiling_config)
            encode_video(video=video, fps=int(frame_rate), audio=None,
                         output_path=base_path, video_chunks_number=chunks)
            del video, audio_out
            gc.collect()
            torch.cuda.empty_cache()
            _gpu_mem("after pass 1 cleanup")

            if use_audio:
                # ============================================================
                # PASS 2: LipDub — redub with lip sync (preserves identity)
                # LipDubPipeline uses IC-LoRA trained to change dialogue
                # while preserving the speaker's appearance and scene.
                # ============================================================
                # Preprocess audio: stereo WAV @16kHz with 0.3s silence padding
                stereo_audio = "/tmp/audio_stereo_16k.wav"
                subprocess.run([
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=stereo",
                    "-t", "0.3",
                    "/tmp/silence_300ms.wav"
                ], capture_output=True, check=True)
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", "/tmp/silence_300ms.wav",
                    "-i", str(audio),
                    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
                    "-ar", "16000", "-ac", "2", "-sample_fmt", "s16",
                    stereo_audio
                ], capture_output=True, check=True)
                print(f"  Audio preprocessed: padded 0.3s + stereo @16kHz", flush=True)

                # Mux audio into base video to create reference video
                ref_video = "/tmp/ref_video_with_audio.mp4"
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", base_path,
                    "-i", stereo_audio,
                    "-c:v", "copy", "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest",
                    ref_video
                ], capture_output=True, check=True)
                print(f"  Reference video created: {ref_video}", flush=True)

                # Free DistilledPipeline, load LipDubPipeline
                print(f"=== PASS 2: Loading LipDubPipeline ===", flush=True)
                del self.pipeline
                gc.collect()
                torch.cuda.empty_cache()
                _gpu_mem("after DistilledPipeline freed")

                from ltx_pipelines.lipdub import LipDubPipeline
                from ltx_core.loader import LoraPathStrengthAndSDOps
                from ltx_core.quantization.fp8_cast import build_policy as fp8_build_policy

                quantization = fp8_build_policy(FP8_CHECKPOINT)
                lipdub_pipeline = LipDubPipeline(
                    distilled_checkpoint_path=FP8_CHECKPOINT,
                    spatial_upsampler_path=SPATIAL_UPSCALER,
                    gemma_root=GEMMA_ROOT,
                    ic_lora=LoraPathStrengthAndSDOps(
                        path=LIPDUB_LORA, strength=1.0, sd_ops=None,
                    ),
                    quantization=quantization,
                )
                _gpu_mem("after LipDubPipeline loaded")

                print(f"=== PASS 2: Running LipDub ===", flush=True)
                video, audio_out = lipdub_pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    images=images,
                    reference_video_path=ref_video,
                    reference_strength=1.0,
                    tiling_config=self.tiling_config,
                    enhance_prompt=enhance_prompt,
                )
                del lipdub_pipeline
                gc.collect()
                torch.cuda.empty_cache()

                # Encode lip-dubbed video (no audio — we'll mux clean audio later)
                encode_video(video=video, fps=int(frame_rate), audio=None,
                             output_path=base_path, video_chunks_number=chunks)
                del video, audio_out
                gc.collect()
                torch.cuda.empty_cache()
                _gpu_mem("after pass 2 complete")

                # ============================================================
                # PASS 3: LatentSync — post-process lip sync correction
                # LatentSync only touches the lip/mouth region,
                # preserving the character's face, hair, and background.
                # It runs on Replicate as a separate API call ($0.10/run).
                # ============================================================
                print(f"=== PASS 3: LatentSync lip sync correction ===", flush=True)

                # First mux clean audio into the lipdub video for LatentSync input
                lipdub_with_audio = "/tmp/lipdub_with_audio.mp4"
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", base_path,
                    "-i", stereo_audio,
                    "-c:v", "copy", "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest",
                    lipdub_with_audio
                ], capture_output=True, check=True)

                try:
                    import requests as _req

                    # Upload video to Replicate file storage
                    with open(lipdub_with_audio, "rb") as f:
                        vid_resp = _req.post(
                            "https://api.replicate.com/v1/files",
                            headers={"Authorization": f"Bearer {os.environ.get('REPLICATE_API_TOKEN', 'os.environ.get('REPLICATE_API_TOKEN', '')')}"},
                            files={"content": ("input.mp4", f, "video/mp4")}
                        )
                    vid_url = vid_resp.json().get("urls", {}).get("get")
                    if not vid_url:
                        raise ValueError(f"Failed to upload video: {vid_resp.text[:200]}")
                    print(f"  Uploaded video for LatentSync", flush=True)

                    # Upload audio
                    with open(stereo_audio, "rb") as f:
                        aud_resp = _req.post(
                            "https://api.replicate.com/v1/files",
                            headers={"Authorization": f"Bearer {os.environ.get('REPLICATE_API_TOKEN', 'os.environ.get('REPLICATE_API_TOKEN', '')')}"},
                            files={"content": ("audio.wav", f, "audio/wav")}
                        )
                    aud_url = aud_resp.json().get("urls", {}).get("get")
                    if not aud_url:
                        raise ValueError(f"Failed to upload audio: {aud_resp.text[:200]}")
                    print(f"  Uploaded audio for LatentSync", flush=True)

                    # Call LatentSync
                    ls_resp = _req.post(
                        "https://api.replicate.com/v1/predictions",
                        headers={
                            "Authorization": f"Bearer {os.environ.get('REPLICATE_API_TOKEN', 'os.environ.get('REPLICATE_API_TOKEN', '')')}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "version": "637ce1919f807ca20da3a448ddc2743535d2853649574cd52a933120e9b9e293",
                            "input": {
                                "video": vid_url,
                                "audio": aud_url,
                                "guidance_scale": lip_sync_strength,
                            }
                        }
                    )
                    ls_data = ls_resp.json()
                    ls_id = ls_data.get("id")
                    if not ls_id:
                        raise ValueError(f"LatentSync prediction failed: {ls_data.get('detail', ls_data.get('error', str(ls_data)[:200]))}")
                    print(f"  LatentSync prediction started: {ls_id}", flush=True)

                    # Poll for completion
                    ls_start = time.time()
                    while True:
                        time.sleep(10)
                        poll = _req.get(
                            f"https://api.replicate.com/v1/predictions/{ls_id}",
                            headers={"Authorization": f"Bearer {os.environ.get('REPLICATE_API_TOKEN', 'os.environ.get('REPLICATE_API_TOKEN', '')')}"},
                        )
                        poll_data = poll.json()
                        status = poll_data.get("status")
                        elapsed = time.time() - ls_start
                        if status == "succeeded":
                            ls_output = poll_data.get("output")
                            if isinstance(ls_output, list):
                                ls_output = ls_output[0] if ls_output else None
                            print(f"  LatentSync completed in {elapsed:.0f}s", flush=True)
                            break
                        elif status == "failed":
                            error = poll_data.get("error", "unknown")
                            print(f"  LatentSync failed: {error[:200]}", flush=True)
                            break
                        elif elapsed > 300:
                            print(f"  LatentSync timeout ({elapsed:.0f}s)", flush=True)
                            break
                        else:
                            print(f"  LatentSync: {status} ({elapsed:.0f}s)...", flush=True)

                    # Download LatentSync output
                    if ls_output and status == "succeeded":
                        subprocess.run([
                            "curl", "-s", "-L", "-o", base_path, ls_output
                        ], capture_output=True, check=True)
                        print(f"  Downloaded LatentSync output", flush=True)
                    else:
                        print(f"  Falling back to LipDub output (no LatentSync)", flush=True)

                except Exception as e:
                    print(f"  LatentSync error: {e}, falling back to LipDub output", flush=True)

        _gpu_mem("after all generation")

        # Frame interpolation with RIFE if target_fps > frame_rate
        output_path = base_path
        if target_fps > 0 and target_fps > frame_rate:
            print(f"=== RIFE interpolation: {int(frame_rate)}fps → {target_fps}fps ===", flush=True)
            rife_start = time.time()
            try:
                output_path = self._interpolate_frames(base_path, target_fps)
                rife_time = time.time() - rife_start
                print(f"=== RIFE done in {rife_time:.0f}s ===", flush=True)
            except Exception as e:
                print(f"RIFE failed ({e}), falling back to base video", flush=True)
                output_path = base_path
        else:
            output_path = base_path

        # Final audio mux (for RIFE or if LatentSync didn't include audio)
        if use_audio and output_path != base_path:
            final_muxed = "/tmp/final_with_audio.mp4"
            r = subprocess.run([
                "ffmpeg", "-y",
                "-i", output_path,
                "-i", stereo_audio if use_audio else "/dev/null",
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                final_muxed
            ], capture_output=True, text=True)
            if r.returncode == 0:
                output_path = final_muxed
                print(f"  Muxed clean audio into final output", flush=True)
            else:
                print(f"  Final audio mux failed: {r.stderr[:200]}", flush=True)
        elif use_audio and output_path == base_path:
            # Mux audio if no RIFE was applied
            muxed = "/tmp/final_with_audio.mp4"
            r = subprocess.run([
                "ffmpeg", "-y",
                "-i", base_path,
                "-i", stereo_audio,
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",
                muxed
            ], capture_output=True, text=True)
            if r.returncode == 0:
                output_path = muxed
                print(f"  Muxed clean audio into output", flush=True)
            else:
                print(f"  Audio mux failed: {r.stderr[:200]}", flush=True)

        elapsed = time.time() - start
        sz = os.path.getsize(output_path) / (1024*1024)
        print("=== DONE: {:.1f}MB in {:.0f}s ===".format(sz, elapsed), flush=True)

        gc.collect()
        torch.cuda.empty_cache()

        return Path(output_path)
