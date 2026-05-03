"""Cog Predictor for sensenova/SenseNova-U1-8B-MoT (text-to-image mode).

Wraps the upstream `sensenova_u1` package (vendored next to this file) which
registers the custom NEO-Unify config/model with `transformers.Auto*`, then
calls `model.t2i_generate` (same path as upstream `examples/t2i/inference.py`).

Weights are staged on R2 (`gs://replicate-weights/lucataco/sensenova-u1-8b-mot/`)
and pulled via `pget` for fast cold boot.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path as _Path
from typing import Any

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

# Make the vendored sensenova_u1 package importable.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import sensenova_u1  # noqa: E402  (registers NEO* with Auto*)
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

MODEL_CACHE = "checkpoints"
WEIGHTS_URL = "https://weights.replicate.delivery/default/lucataco/sensenova-u1-8b-mot/model.tar"

# Native training resolution buckets (from upstream examples/t2i/inference.py).
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1 (2048x2048)": (2048, 2048),
    "16:9 (2720x1536)": (2720, 1536),
    "9:16 (1536x2720)": (1536, 2720),
    "3:2 (2496x1664)": (2496, 1664),
    "2:3 (1664x2496)": (1664, 2496),
    "4:3 (2368x1760)": (2368, 1760),
    "3:4 (1760x2368)": (1760, 2368),
    "2:1 (2880x1440)": (2880, 1440),
    "1:2 (1440x2880)": (1440, 2880),
}

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def _coerce(val, expected_type, default):
    return val if isinstance(val, expected_type) else default


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def download_weights(url: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    t0 = time.time()
    print(f"[setup] pget {url} -> {dest}")
    subprocess.check_call(
        ["pget", "--log-level", "warn", "-f", "-x", url, dest],
        close_fds=False,
    )
    print(f"[setup] pget done in {time.time() - t0:.1f}s")


class Predictor(BasePredictor):
    def setup(self) -> None:
        marker = os.path.join(MODEL_CACHE, "config.json")
        if not os.path.exists(marker):
            download_weights(WEIGHTS_URL, MODEL_CACHE)

        # Force SDPA (no flash-attn in container by default).
        sensenova_u1.set_attn_backend("sdpa")
        print(f"[setup] attn backend: {sensenova_u1.effective_attn_backend()}")

        config = AutoConfig.from_pretrained(MODEL_CACHE)
        sensenova_u1.check_checkpoint_compatibility(config)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE)
        self.model = (
            AutoModel.from_pretrained(MODEL_CACHE, config=config, torch_dtype=torch.bfloat16)
            .to("cuda")
            .eval()
        )
        if torch.cuda.is_available():
            print(
                f"[setup] cuda mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the image to generate.",
            default="A vibrant male peacock with his tail feathers fully fanned out in a wide, iridescent display of blues and greens, standing in a garden.",
        ),
        aspect_ratio: str = Input(
            description="Output resolution bucket. SenseNova U1 is trained on ~2K resolutions.",
            default="1:1 (2048x2048)",
            choices=list(RESOLUTIONS.keys()),
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps.",
            default=50,
            ge=10,
            le=100,
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale.",
            default=4.0,
            ge=1.0,
            le=15.0,
        ),
        timestep_shift: float = Input(
            description="Timestep schedule shift. Higher = more emphasis on early/coarse steps.",
            default=3.0,
            ge=0.5,
            le=8.0,
        ),
        seed: int = Input(
            description="Random seed. Set -1 to randomize.",
            default=-1,
        ),
    ) -> Path:
        # Defend against coglet 0.19.x FieldInfo leak on omitted optional inputs.
        prompt = _coerce(prompt, str, "")
        aspect_ratio = _coerce(aspect_ratio, str, "1:1 (2048x2048)")
        num_inference_steps = _coerce(num_inference_steps, int, 50)
        cfg_scale = _coerce(cfg_scale, (int, float), 4.0)
        timestep_shift = _coerce(timestep_shift, (int, float), 3.0)
        seed = _coerce(seed, int, -1)
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
        print(f"[predict] seed={seed}")

        width, height = RESOLUTIONS.get(aspect_ratio, (2048, 2048))

        t0 = time.time()
        out = self.model.t2i_generate(
            self.tokenizer,
            prompt,
            image_size=(width, height),
            cfg_scale=float(cfg_scale),
            cfg_norm="none",
            timestep_shift=float(timestep_shift),
            cfg_interval=(0.0, 1.0),
            num_steps=int(num_inference_steps),
            batch_size=1,
            seed=int(seed),
            think_mode=False,
        )
        tensor = out[0] if isinstance(out, tuple) else out
        elapsed = time.time() - t0
        print(f"[predict] generation: {elapsed:.1f}s ({width}x{height}, {num_inference_steps} steps)")

        images = _to_pil(tensor)
        out_path = "/tmp/output.png"
        images[0].save(out_path)
        return Path(out_path)
