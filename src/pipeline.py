"""
AI Video Generation Pipeline — End-to-End Flow
================================================
Orchestrates the complete video generation pipeline:

1. Prompt → LTX-2.3 video generation (Replicate/GPU)
2. Optional: LipDub pass for talking-head videos
3. Optional: LatentSync lip sync correction
4. Optional: RIFE frame interpolation (24fps → 60fps)
5. Audio mixing: dialogue + SFX + background music with sidechain ducking
6. Final encode and delivery

Usage:
  python pipeline.py --prompt "A cat on a windowsill" --output final.mp4
  python pipeline.py --prompt "Product demo" --image ref.png --audio voice.wav --output final.mp4
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Optional


class VideoPipeline:
    """
    End-to-end video generation pipeline.

    Stages:
    1. Generate base video (LTX-2.3 on GPU)
    2. Optional: LipDub for dialogue replacement
    3. Optional: LatentSync for lip sync correction
    4. Optional: RIFE frame interpolation for smooth playback
    5. Mix audio (dialogue + SFX + music with ducking)
    6. Final encode with H.264 + AAC
    """

    def __init__(self, replicate_token: str, hf_token: str):
        self.replicate_token = replicate_token
        self.hf_token = hf_token

    def generate_video(
        self,
        prompt: str,
        output_path: str,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        width: int = 768,
        height: int = 1344,
        num_frames: int = 97,
        frame_rate: float = 24.0,
        seed: int = -1,
        target_fps: int = 60,
        lip_sync_strength: float = 1.5,
    ) -> str:
        """
        Run the full pipeline. Returns path to final video.

        The actual generation runs on Replicate (cloud GPU) using the
        LTX-2.3 cog predictor (see src/predict.py).
        """
        import requests

        print("=" * 60)
        print("🎬 AI VIDEO GENERATION PIPELINE")
        print("=" * 60)

        # Step 1: Submit prediction to Replicate
        print("\n📍 Step 1: Submitting to Replicate (LTX-2.3)...")
        start = time.time()

        prediction = self._submit_prediction(
            prompt=prompt,
            image=image,
            audio=audio,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            target_fps=target_fps,
            lip_sync_strength=lip_sync_strength,
        )

        pred_id = prediction.get("id")
        print(f"  Prediction ID: {pred_id}")

        # Step 2: Poll for completion
        print("\n📍 Step 2: Waiting for generation (this takes 2-5 min)...")
        result_url = self._poll_prediction(pred_id)
        gen_time = time.time() - start
        print(f"  ✅ Generated in {gen_time:.0f}s")

        # Step 3: Download result
        print("\n📍 Step 3: Downloading video...")
        raw_path = "/tmp/pipeline_raw.mp4"
        subprocess.run(["curl", "-s", "-L", "-o", raw_path, result_url], check=True)
        print(f"  ✅ Downloaded: {os.path.getsize(raw_path) / (1024*1024):.1f}MB")

        # Step 4: Audio mixing (if audio provided or SFX needed)
        if audio:
            print("\n📍 Step 4: Mixing audio...")
            final_path = self._mix_audio(raw_path, audio, output_path)
        else:
            # Just copy to output
            subprocess.run(["cp", raw_path, output_path])
            final_path = output_path

        # Cleanup
        if os.path.exists(raw_path) and raw_path != output_path:
            os.unlink(raw_path)

        total_time = time.time() - start
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"\n✅ DONE: {final_path} ({size_mb:.1f}MB, {total_time:.0f}s total)")

        return final_path

    def _submit_prediction(self, **kwargs) -> dict:
        """Submit prediction to Replicate API."""
        import requests

        # The model version hash for our LTX-2.3 cog deployment
        MODEL_VERSION = "ltx-2.3-cog"  # Replace with actual version hash

        # Build input
        input_data = {
            "prompt": kwargs["prompt"],
            "hf_token": self.hf_token,
            "width": kwargs["width"],
            "height": kwargs["height"],
            "num_frames": kwargs["num_frames"],
            "frame_rate": kwargs["frame_rate"],
            "seed": kwargs["seed"],
        }

        if kwargs.get("image"):
            # Upload image to Replicate file storage
            with open(kwargs["image"], "rb") as f:
                upload = requests.post(
                    "https://api.replicate.com/v1/files",
                    headers={"Authorization": f"Bearer {self.replicate_token}"},
                    files={"content": f},
                )
            input_data["image"] = upload.json().get("urls", {}).get("get")

        if kwargs.get("audio"):
            with open(kwargs["audio"], "rb") as f:
                upload = requests.post(
                    "https://api.replicate.com/v1/files",
                    headers={"Authorization": f"Bearer {self.replicate_token}"},
                    files={"content": f},
                )
            input_data["audio"] = upload.json().get("urls", {}).get("get")
            input_data["lip_sync_strength"] = kwargs["lip_sync_strength"]

        if kwargs.get("target_fps", 0) > 0:
            input_data["target_fps"] = kwargs["target_fps"]

        resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Bearer {self.replicate_token}",
                "Content-Type": "application/json",
            },
            json={"version": MODEL_VERSION, "input": input_data},
        )
        return resp.json()

    def _poll_prediction(self, pred_id: str, timeout: int = 600) -> str:
        """Poll Replicate for prediction completion."""
        import requests

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(10)
            resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{pred_id}",
                headers={"Authorization": f"Bearer {self.replicate_token}"},
            )
            data = resp.json()
            status = data.get("status")
            elapsed = time.time() - start

            if status == "succeeded":
                output = data.get("output")
                if isinstance(output, list):
                    output = output[0] if output else None
                return output
            elif status == "failed":
                raise RuntimeError(f"Generation failed: {data.get('error', 'unknown')}")
            else:
                print(f"  ... {status} ({elapsed:.0f}s)")

        raise TimeoutError(f"Prediction timed out after {timeout}s")

    def _mix_audio(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Run audio mixing pipeline (delegates to mix_audio.py)."""
        mixer = os.path.join(os.path.dirname(__file__), "..", "audio", "mix_audio.py")
        if not os.path.exists(mixer):
            mixer = os.path.join(os.path.dirname(__file__), "audio", "mix_audio.py")

        cmd = [
            sys.executable, mixer,
            "--video", video_path,
            "--output", output_path,
        ]
        subprocess.run(cmd, check=True)
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Video Generation Pipeline")
    parser.add_argument("--prompt", required=True, help="Text prompt for video")
    parser.add_argument("--image", default=None, help="Reference image for img2vid")
    parser.add_argument("--audio", default=None, help="Audio file for lipsync")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--target-fps", type=int, default=60)
    parser.add_argument("--lip-sync-strength", type=float, default=1.5)

    args = parser.parse_args()

    replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not replicate_token or not hf_token:
        print("Error: Set REPLICATE_API_TOKEN and HF_TOKEN environment variables")
        sys.exit(1)

    pipeline = VideoPipeline(replicate_token, hf_token)
    pipeline.generate_video(
        prompt=args.prompt,
        output_path=args.output,
        image=args.image,
        audio=args.audio,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        target_fps=args.target_fps,
        lip_sync_strength=args.lip_sync_strength,
    )
