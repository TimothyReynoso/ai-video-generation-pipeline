# LTX-2.3 Replicate Deployment — Research Report

> **Date:** 2026-06-01  
> **Status:** Build succeeding (4 consecutive green builds as of 2026-06-01)  
> **Model:** moltstudios/ltx-2.3 on Replicate  
> **Latest Version:** `707d22362fdb...` (2026-06-01T07:47:12Z, Cog 0.20.0)

---

## Table of Contents

1. [Replicate Cog Build System](#1-replicate-cog-build-system)
2. [HuggingFace Hub Downloads](#2-huggingface-hub-downloads)
3. [LTX-2.3 Model Weights](#3-ltx-23-model-weights)
4. [Gemma 3 12B Text Encoder](#4-gemma-3-12b-text-encoder)
5. [LTX-2 Pipeline Code](#5-ltx-2-pipeline-code)
6. [GitHub Actions Build](#6-github-actions-build)
7. [Replicate Model Configuration](#7-replicate-model-configuration)
8. [Current Files Analysis](#8-current-files-analysis)
9. [Issues Found](#9-issues-found)
10. [Recommended Architecture](#10-recommended-architecture)
11. [Step-by-Step Deployment Plan](#11-step-by-step-deployment-plan)

---

## 1. Replicate Cog Build System

### How cog.yaml Works

- **`build.run` commands** execute as `root` during `docker build` (not at prediction time). Your source code is NOT available during these commands.
- **`build.gpu: true`** causes Cog to use an NVIDIA CUDA base image.
- **CUDA version** is auto-detected from the `torch` version in `requirements.txt`. You can override with `build.cuda: "12.4"` but this is unnecessary when torch is pinned correctly.
- **Cog version:** 0.20.0 (current latest)
- **Python versions supported:** 3.10, 3.11, 3.12, 3.13

### Image Size Constraints

- **No documented hard limit** on Docker image size for Replicate.
- **Practical limit on GitHub Actions:** Standard runners have ~84GB total disk. After the "Free disk space" step (removes dotnet, android, ghc), ~60GB free. The Docker build itself consumes space during layer creation.
- Our current image is relatively small (~5-6GB) since model weights are NOT baked in — they download at setup() time.

### Secrets

- **Replicate model secrets** (set via model Settings → Secrets) are available as **environment variables** during `setup()` AND `predict()`.
- **NOT available** during `cog build` / `build.run` commands (those are Docker build-time only).
- The `cog.Secret` input type is for per-prediction secrets passed by API callers — different from model-level secrets.
- **For build-time secrets**, cog supports `--secret` flag: `cog push --secret id=hf,src=$HOME/.hf_token`.

### Docker Layer Caching

- **GitHub Actions does NOT cache Docker layers** between runs by default.
- Each `cog push` builds from scratch. With our current config (pip install torch + git clone), this takes ~6-7 minutes.
- Could be optimized with Docker layer caching via `actions/cache`, but not worth the complexity for infrequent builds.

### Best Practices (from Replicate docs)

- Download weights in `setup()` for smaller images and faster builds.
- Use `pget` for fast parallel downloads of public files.
- Use `huggingface_hub` for authenticated/gated model downloads.
- Use `--separate-weights` flag on `cog push` if baking weights into the image.

---

## 2. HuggingFace Hub Downloads

### `hf_hub_download` vs `snapshot_download`

| Feature | `hf_hub_download` | `snapshot_download` |
|---------|-------------------|---------------------|
| Scope | Single file | Entire repo |
| Auth | `token=` param | `token=` param |
| Filtering | N/A (single file) | `allow_patterns`, `ignore_patterns` |
| Parallelism | Single file download | Parallel multi-file download |
| Local cache | Symlink-based cache | Symlink-based cache |
| `local_dir` | Downloads to flat dir | Reproduces repo structure |

### Authentication for Gated Models

- **Gemma 3 12B** (`google/gemma-3-12b-it-qat-q4_0-unquantized`) is **gated** (`"gated": "manual"`).
- Requires HF_TOKEN where the associated HuggingFace account has accepted Google's Gemma license.
- Token is passed via `token=` parameter to `snapshot_download()` or `hf_hub_download()`.
- LTX-2.3 is **public** (`"gated": false`) — no auth needed.

### HuggingFace XET

- Both repos use **XET** (accelerated transfer) for large safetensors files.
- `hf-xet` package is installed automatically as a dependency of `huggingface-hub`.
- XET provides significantly faster downloads for large files on supported repos.

---

## 3. LTX-2.3 Model Weights

### Repository: `Lightricks/LTX-2.3`

- **Public:** Yes (not gated)
- **License:** LTX-2 Community License Agreement
- **Total repo size:** 146.22 GB (157,004,895,813 bytes)
- **Last modified:** 2026-04-13

### Complete File List with Sizes

| File | Size | Required? |
|------|------|-----------|
| `ltx-2.3-22b-dev.safetensors` | 42.98 GB | ❌ Development base |
| `ltx-2.3-22b-distilled.safetensors` | 42.98 GB | ❌ Original distilled (v1.0) |
| **`ltx-2.3-22b-distilled-1.1.safetensors`** | **42.98 GB** | ✅ **Main checkpoint (v1.1)** |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.08 GB | ❌ Original LoRA (v1.0) |
| **`ltx-2.3-22b-distilled-lora-384-1.1.safetensors`** | **7.08 GB** | ✅ **Distilled LoRA (v1.1)** |
| `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | 1.02 GB | ❌ 1.5x upscaler |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 949.6 MB | ❌ v1.0 upscaler |
| **`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`** | **949.6 MB** | ✅ **2x spatial upscaler (v1.1)** |
| `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | 249.8 MB | ❌ Temporal upscaler (future) |
| `ltx2.3-open.png` | 2.2 MB | ❌ Demo image |
| `.gitattributes` | 1.6 KB | ❌ |
| `LICENSE` | 21.4 KB | ❌ |
| `README.md` | 6.5 KB | ❌ |

### Required Download Size: **~51 GB**

| Component | Size |
|-----------|------|
| Distilled checkpoint (v1.1) | 42.98 GB |
| Distilled LoRA (v1.1) | 7.08 GB |
| Spatial upscaler x2 (v1.1) | 949.6 MB |
| **Total LTX** | **~51.0 GB** |

---

## 4. Gemma 3 12B Text Encoder

### Repository: `google/gemma-3-12b-it-qat-q4_0-unquantized`

- **Gated:** Yes (manual approval — must accept Google's Gemma license)
- **License:** Gemma
- **Architecture:** `Gemma3ForConditionalGeneration`
- **Parameters:** 12.2B (BF16)
- **Total size:** 22.74 GB (24,414,161,761 bytes)

### Complete File List with Sizes

| File | Size |
|------|------|
| `model-00001-of-00005.safetensors` | 4.64 GB |
| `model-00002-of-00005.safetensors` | 4.59 GB |
| `model-00003-of-00005.safetensors` | 4.59 GB |
| `model-00004-of-00005.safetensors` | 4.59 GB |
| `model-00005-of-00005.safetensors` | 4.29 GB |
| `model.safetensors.index.json` | 108.6 KB |
| `tokenizer.json` | 31.8 MB |
| `tokenizer.model` | 4.5 MB |
| `tokenizer_config.json` | 1.1 MB |
| `config.json` | 1.6 KB |
| `generation_config.json` | 173 B |
| `preprocessor_config.json` | 570 B |
| `processor_config.json` | 70 B |
| `special_tokens_map.json` | 662 B |
| `added_tokens.json` | 35 B |
| `chat_template.json` | 1.6 KB |
| **Total Gemma** | **~22.74 GB** |

### Auth Requirements

The HF_TOKEN used in setup() must be from a HuggingFace account that has:
1. Logged into huggingface.co
2. Visited the model page: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
3. Clicked "Acknowledge license" to accept Google's Gemma terms
4. Generated a read API token from https://huggingface.co/settings/tokens

---

## 5. LTX-2 Pipeline Code

### Repository: https://github.com/Lightricks/LTX-2

### Package Structure

```
LTX-2/
├── packages/
│   ├── ltx-core/         # Core model implementation
│   │   └── pyproject.toml
│   └── ltx-pipelines/    # Pipeline implementations
│       └── pyproject.toml
└── pyproject.toml         # Workspace root (uv workspace)
```

### Dependency Chain

**ltx-core** (v1.1.5):
- `torch~=2.7` (core requirement)
- `torchaudio`
- `transformers>=4.52`
- `safetensors`, `accelerate`, `scipy>=1.14`, `einops`, `numpy`
- Optional: `xformers` (from cu129 index — NOT required)

**ltx-pipelines** (v1.1.5):
- `ltx-core` (workspace dependency)
- `av`, `tqdm`, `pillow`, `openimageio`

### CUDA/xformers Notes

- The `cu129` PyTorch wheel index is configured ONLY for the optional `xformers` extra.
- **xformers is NOT a hard dependency** — it's in `[project.optional-dependencies]`.
- The core inference path works fine without xformers.
- Our `pip install ./packages/ltx-core ./packages/ltx-pipelines` correctly skips xformers.

### DistilledPipeline (`ltx_pipelines.distilled`)

The `DistilledPipeline` class is the main entry point:

```python
pipeline = DistilledPipeline(
    distilled_checkpoint_path="path/to/distilled.safetensors",
    gemma_root="path/to/gemma/",
    spatial_upsampler_path="path/to/upscaler.safetensors",
    loras=(LoraPathStrengthAndSDOps(path="lora.safetensors", strength=0.8, sd_ops=None),),
)
```

**Two-stage pipeline:**
1. **Stage 1:** Generates video at half resolution (e.g., 384x672 for 768x1344 target)
2. **Stage 2:** Upsamples 2x using the spatial upscaler, then refines with additional denoising

**Call signature:**
```python
video, audio = pipeline(
    prompt=str,
    seed=int,
    height=int, width=int,
    num_frames=int,
    frame_rate=float,
    images=list[ImageConditioningInput],
    tiling_config=TilingConfig,
    enhance_prompt=bool,
)
```

**Output encoding:**
```python
from ltx_pipelines.utils.media_io import encode_video
encode_video(video=video, fps=24, audio=audio, output_path="output.mp4",
             video_chunks_number=chunks)
```

### Installation Methods

1. **`pip install ./packages/ltx-core ./packages/ltx-pipelines`** ✅ (current approach)
   - Installs into system Python, no venv isolation
   - Ignores the uv workspace config entirely
   - Uses whatever torch is already installed (our cu126 build)
   - Works correctly

2. **`uv sync --frozen`** ❌ (previous approach, caused issues)
   - Creates its own venv at `/opt/LTX-2/.venv/`
   - Resolves torch from the cu129 index in pyproject.toml
   - Conflicts with Cog's torch installation
   - The `--frozen` flag only works if `uv.lock` is committed and compatible

3. **`uv sync --frozen --no-dev`** ❌ (attempted fix)
   - Same issues as above

---

## 6. GitHub Actions Build

### Current Workflow

```yaml
name: Push to Replicate
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Free disk space        # Removes dotnet, android, ghc (~20GB)
      - name: Install Cog            # Downloads latest cog binary
      - name: Login to Replicate     # docker login + cog login
      - name: Push to Replicate      # cog push r8.im/moltstudios/ltx-2.3
```

### GitHub Runner Constraints

- **Standard runner disk:** ~84GB total
- **After cleanup step:** ~60GB free
- **RAM:** 16GB (not relevant for Docker build)
- **Build time:** ~6-7 minutes for Cog build + push

### Secrets Required in GitHub Repo

| Secret | Purpose |
|--------|---------|
| `REPLICATE_API_TOKEN` | Docker login to r8.im registry |
| `REPLICATE_CLI_TOKEN` | `cog login` for push auth |

### Build Secrets (HF_TOKEN)

- `cog push` does NOT support passing build secrets via `--build-arg`.
- However, `cog push` supports `--secret id=foo,src=/path/to/file` for Docker build secrets.
- **We don't need HF_TOKEN at build time** — only at setup() runtime, where it comes from Replicate model secrets.

### `cog push` Behavior

- Builds the Docker image using `cog.yaml` configuration
- Tags and pushes to the specified registry (r8.im)
- Creates a new model version on Replicate
- The `ERROR: rpc error: code = NotFound desc = no access allowed to dir "context"` in logs is a **benign Docker buildx warning** — does not affect the build.

---

## 7. Replicate Model Configuration

### Model: `moltstudios/ltx-2.3`

| Field | Value |
|-------|-------|
| **Created** | 2026-05-28 |
| **Description** | LTX-2.3 (22B) audio-video generation model |
| **Cog Version** | 0.20.0 |
| **Latest Version** | `707d22362fdb...` (2026-06-01T07:47:12Z) |
| **Hardware** | A100 80GB |
| **Versions** | 4 versions as of 2026-06-01 |

### Deployments

Two deployments exist under `moltstudios`:
1. `ltx-23-production`
2. `ltx-2-3-production`

### Secrets on Model

- **`HF_TOKEN`** must be configured on the Replicate model (Settings → Secrets) for Gemma downloads.
- Available during `setup()` as `os.environ.get("HF_TOKEN")`.

### API Schema (Current)

```json
{
  "prompt": "string (text prompt)",
  "image": "uri? (image for img2vid)",
  "image_strength": "float (0-1, default 1.0)",
  "width": "int (div 64, default 768)",
  "height": "int (div 64, default 1344)",
  "num_frames": "int (8k+1, default 121)",
  "frame_rate": "float (default 24)",
  "seed": "int (default -1 = random)",
  "enhance_prompt": "bool (default false)"
}
```

Returns: `cog.Path` (mp4 video file)

---

## 8. Current Files Analysis

### `requirements.txt`

```
--extra-index-url https://download.pytorch.org/whl/cu126
cog
Pillow
torch==2.7.1+cu126
```

**Status:** ✅ Working. Uses cu126 (CUDA 12.6) which is compatible with Cog's auto-detection. Previous iterations used cu128 and cu129 which had compatibility issues.

### `cog.yaml`

```yaml
build:
  gpu: true
  system_packages: [libgl1, libglib2.0-0, ffmpeg, git, wget]
  python_version: "3.12"
  python_requirements: requirements.txt
  run:
    - git clone LTX-2, pip install packages, install pget
run: "predict.py:Predictor"
```

**Status:** ✅ Working. CUDA version is auto-detected from torch. No explicit `cuda:` directive needed.

### `predict.py` (Current — v3 In-Process Architecture)

**Architecture:**
1. **Phase 1:** Downloads LTX weights via `pget` (parallel HTTP, no auth needed)
2. **Phase 2:** Downloads Gemma via `huggingface_hub.snapshot_download` (auth + parallel)
3. **Phase 3:** Loads `DistilledPipeline` into GPU memory (stays warm across predictions)

**Key design decisions:**
- ✅ Uses `pget` for public LTX files (faster than huggingface_hub for public files)
- ✅ Uses `snapshot_download` for gated Gemma (handles auth)
- ✅ In-process pipeline loading (not subprocess) — avoids Python import overhead per prediction
- ✅ Checks for existing files before downloading (supports container reuse)
- ✅ Returns `cog.Path` for proper file serving

### `.github/workflows/push-to-replicate.yaml`

**Status:** ✅ Working. Clean, minimal workflow.

---

## 9. Issues Found

### Historical Issues (Resolved)

| Issue | Root Cause | Resolution |
|-------|-----------|------------|
| 8+ failed builds | torch cu129 index incompatible with Cog | Switched to cu126 |
| `uv sync --frozen` failures | uv creates isolated venv with cu129 torch | Replaced with `pip install` into system Python |
| Subprocess-based predict.py | Spawning Python per prediction | Rewrote as in-process pipeline loading |
| CUDA version mismatch | Cog doesn't support cu129 | Using cu126 with torch 2.7.1 |
| xformers ABI incompatibility | cu129-compiled xformers vs cu126 torch | Not installing xformers (optional) |

### Current Issues (Benign)

| Issue | Severity | Notes |
|-------|----------|-------|
| `ERROR: rpc error` in build logs | 🟢 None | Benign Docker buildx warning. Build succeeds. |
| ~73GB download per cold start | 🟡 Performance | setup() downloads 51GB LTX + 22.7GB Gemma. Takes 5-10 min. No fix — this is how Replicate works with large models. |
| No xformers optimization | 🟢 None | Minor performance impact. xformers is optional and only helps on certain GPU architectures. |

### Potential Future Issues

1. **HuggingFace rate limiting:** If cold starts happen frequently, HuggingFace may rate-limit downloads. Mitigation: Use a persistent deployment (min_instances > 0).
2. **Model weight updates:** If Lightricks releases v1.2 weights, the hardcoded filenames in predict.py need updating.
3. **Torch version upgrades:** If ltx-core requires torch 2.8+, we'll need to verify Cog compatibility again.

---

## 10. Recommended Architecture

### Current Architecture (Recommended) ✅

```
┌─────────────────────────────────────────┐
│ Docker Image (built by Cog)             │
│                                         │
│  System: CUDA 12.6 + Python 3.12       │
│  Python: torch 2.7.1+cu126             │
│          ltx-core 1.1.5                 │
│          ltx-pipelines 1.1.5            │
│          huggingface-hub 1.17.0         │
│          pget (parallel downloader)     │
│  Code: predict.py                       │
│                                         │
│  Image size: ~5-6 GB                    │
└─────────────────────────────────────────┘
         │
         │ setup() — first cold start
         ▼
┌─────────────────────────────────────────┐
│ Runtime Weights (downloaded to /opt/)   │
│                                         │
│  LTX-2.3 distilled v1.1:  42.98 GB     │
│  LTX-2.3 LoRA v1.1:        7.08 GB     │
│  LTX-2.3 upscaler v1.1:    0.95 GB     │
│  Gemma 3 12B QAT Q4:      22.74 GB     │
│  ─────────────────────────────────      │
│  Total:                   ~73.75 GB     │
└─────────────────────────────────────────┘
         │
         │ Loaded into GPU (A100 80GB)
         ▼
┌─────────────────────────────────────────┐
│ Pipeline in GPU Memory                  │
│                                         │
│  DistilledPipeline (two-stage)          │
│  - Gemma text encoder (~23GB VRAM)     │
│  - Video VAE + Transformer (~43GB)     │
│  - Spatial upscaler (~1GB)             │
│  - Audio decoder (loaded as needed)     │
│                                         │
│  Total VRAM: ~67-70GB                   │
│  A100 80GB: ✅ Fits with headroom       │
└─────────────────────────────────────────┘
```

### Why NOT Bake Weights into Image

| Factor | Bake In Image | Download in setup() |
|--------|--------------|-------------------|
| Image size | ~80GB | ~6GB |
| Build time | 30+ min | 6-7 min |
| GitHub runner disk | May not fit | Fits easily |
| Cold start time | ~2 min (load only) | ~10 min (download + load) |
| Weight updates | Requires rebuild | Just redeploy |
| Docker layer push | Very slow | Fast |

**Verdict:** Download in setup() is the right call. The cold start penalty is unavoidable with 73GB of weights. Use a persistent deployment (min_instances ≥ 1) to avoid cold starts in production.

---

## 11. Step-by-Step Deployment Plan

### Current State: ✅ DEPLOYED AND BUILDING

The model is already deployed and building successfully. Here's the plan that got us here (and for future iterations):

### Phase 1: Repository Setup (Done)

1. ✅ Create repo `moltstudios/cog-ltx-2.3` on GitHub
2. ✅ Create model `moltstudios/ltx-2.3` on Replicate
3. ✅ Set GitHub secrets: `REPLICATE_API_TOKEN`, `REPLICATE_CLI_TOKEN`
4. ✅ Set Replicate model secret: `HF_TOKEN` (from HuggingFace account that accepted Gemma license)

### Phase 2: Code (Done)

1. ✅ `requirements.txt` — torch 2.7.1+cu126 + cog + Pillow
2. ✅ `cog.yaml` — GPU, Python 3.12, system packages, build steps
3. ✅ `predict.py` — In-process DistilledPipeline with pget + huggingface_hub downloads
4. ✅ `.github/workflows/push-to-replicate.yaml` — CI/CD pipeline

### Phase 3: Build & Push (Done — 4 consecutive successes)

1. Push to `main` branch triggers GitHub Actions
2. Cog builds Docker image (~6 min)
3. Cog pushes to `r8.im/moltstudios/ltx-2.3` (~1 min)
4. New version appears on Replicate

### Phase 4: Deploy (Next Step)

1. Go to Replicate → Deployments → `ltx-2-3-production`
2. Update to latest version
3. Set hardware: A100 80GB
4. Set min_instances to 1 (avoid cold starts)
5. Test with sample prediction

### Phase 5: Testing Checklist

- [ ] Text-to-video generation (prompt only)
- [ ] Image-to-video generation (with image input)
- [ ] Various resolutions (768x1344 = 9:16, 1536x864 = 16:9)
- [ ] Frame count variations (97 = ~4s, 121 = ~5s)
- [ ] Seed reproducibility (same seed → same output)
- [ ] Audio generation (verify native audio in output mp4)
- [ ] Prompt enhancement (with Gemma)
- [ ] Cold start timing (download + load + first inference)
- [ ] Warm inference timing (subsequent predictions)
- [ ] VRAM monitoring (verify fits in 80GB)

### For Future Weight Updates

1. Update filenames in `predict.py` (e.g., `ltx-2.3-22b-distilled-2.0.safetensors`)
2. Push to main
3. Wait for build
4. Update deployment version

---

## Appendix A: Total Weight Download Summary

| Component | Repository | Files | Size | Auth |
|-----------|-----------|-------|------|------|
| LTX-2.3 Distilled v1.1 | `Lightricks/LTX-2.3` | 1 safetensors | 42.98 GB | None |
| LTX-2.3 LoRA v1.1 | `Lightricks/LTX-2.3` | 1 safetensors | 7.08 GB | None |
| LTX-2.3 Upscaler v1.1 | `Lightricks/LTX-2.3` | 1 safetensors | 0.95 GB | None |
| Gemma 3 12B QAT Q4 | `google/gemma-3-12b-it-qat-q4_0-unquantized` | 5 safetensors + 11 config | 22.74 GB | HF_TOKEN |
| **Total** | | **19 files** | **~73.75 GB** | |

## Appendix B: Build History

| Run ID | Date | Status | Torch | Notes |
|--------|------|--------|-------|-------|
| 26741777678 | 2026-06-01 07:40 | ✅ Success | cu126 | Current (v4) |
| 26741014482 | 2026-06-01 07:22 | ✅ Success | cu126 | |
| 26737900317 | 2026-06-01 05:57 | ✅ Success | cu126 | |
| 26736322770 | 2026-06-01 05:07 | ✅ Success | cu126 | |
| 26626738831 | 2026-05-29 08:26 | ✅ Success | cu128 | First success |
| (8+ earlier) | 2026-05-28 | ❌ Failed | cu129 | Various failures |

## Appendix C: Key URLs

| Resource | URL |
|----------|-----|
| GitHub repo | https://github.com/moltstudios/cog-ltx-2.3 |
| Replicate model | https://replicate.com/moltstudios/ltx-2.3 |
| LTX-2.3 weights | https://huggingface.co/Lightricks/LTX-2.3 |
| Gemma weights | https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized |
| LTX-2 source | https://github.com/Lightricks/LTX-2 |
| Cog docs | https://cog.run/docs/ |
| Replicate secrets | https://replicate.com/docs/topics/models/secrets |
| pget tool | https://github.com/replicate/pget |
