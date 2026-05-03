# cog-sensenova-u1-8b-mot

[![Replicate](https://replicate.com/lucataco/sensenova-u1-8b-mot/badge)](https://replicate.com/lucataco/sensenova-u1-8b-mot)

Cog wrapper for [`sensenova/SenseNova-U1-8B-MoT`](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) — a unified multimodal model from SenseNova that natively handles text-to-image, image editing, VQA, and interleaved text+image generation via the [NEO-Unify](https://huggingface.co/blog/sensenova/neo-unify) architecture (no separate vision encoder, no VAE).

This wrapper exposes the **text-to-image** path on Replicate. Editing / VQA / interleaved generation may follow as separate models.

## Architecture

- 8B dense backbone — Qwen3 LLM (42 layers, 4096 hidden) with a native NEO vision tower (1024 hidden, 16x16 patch).
- **No VAE / no separate vision encoder** — text and pixel information modelled end-to-end as a unified compound (Mixture-of-Tokens).
- Trained at ~2K resolution buckets (1:1, 16:9, 3:2, 4:3, 2:1, etc.).
- Apache-2.0 license on both code and weights.

## Run on Replicate

[https://replicate.com/lucataco/sensenova-u1-8b-mot](https://replicate.com/lucataco/sensenova-u1-8b-mot)

```python
import replicate

output = replicate.run(
    "lucataco/sensenova-u1-8b-mot",
    input={
        "prompt": "a cute orange tabby kitten in a sunny garden",
        "aspect_ratio": "1:1 (2048x2048)",
        "num_inference_steps": 30,
        "cfg_scale": 4.0,
    },
)
print(output)  # https://replicate.delivery/.../output.png
```

```bash
curl -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lucataco/sensenova-u1-8b-mot",
    "input": {
      "prompt": "a vibrant infographic about coffee brewing methods",
      "aspect_ratio": "16:9 (2720x1536)",
      "num_inference_steps": 50
    }
  }'
```

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | string | (peacock example) | Text prompt describing the image. |
| `aspect_ratio` | choice | `1:1 (2048x2048)` | Native trained resolution buckets. |
| `num_inference_steps` | int (10-100) | 50 | Number of denoising steps. |
| `cfg_scale` | float (1-15) | 4.0 | Classifier-free guidance scale. |
| `timestep_shift` | float (0.5-8) | 3.0 | Schedule shift toward early steps. |
| `seed` | int | -1 (random) | Random seed for reproducibility. |

## Run locally

```bash
git clone https://github.com/lucataco/cog-sensenova-u1-8b-mot
cd cog-sensenova-u1-8b-mot
cog predict -i prompt="a red apple on a wooden table"
```

Requires a GPU with at least ~36 GB VRAM at BF16 (L40S 48GB or larger).

## How it works

- Vendors the upstream [`sensenova_u1`](https://github.com/OpenSenseNova/SenseNova-U1) Python package, which registers the custom `NEOChatModel` / `NEOVisionModel` with `transformers.Auto*` at import time. No `trust_remote_code` needed.
- Calls `model.t2i_generate(...)` directly (the same path as upstream `examples/t2i/inference.py`).
- Weights are pre-staged as a single tar on Replicate's R2 CDN and pulled with `pget` for fast cold boots (~2 min vs ~9 min from HuggingFace).

## License

- **Cog wrapper code**: Apache-2.0 (see `LICENSE`).
- **Model weights**: Apache-2.0, governed by the upstream [SenseNova-U1 model card](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT/blob/main/LICENSE).

## Credits

- [SenseNova / OpenSenseNova](https://github.com/OpenSenseNova/SenseNova-U1) — model and inference code.
- [Replicate](https://replicate.com) — Cog and serving platform.
