# SenseNova U1 8B MoT

SenseNova U1 is a new family of unified multimodal models from SenseNova that handles text and images together as a single, native modality — rather than bolting a vision encoder onto a language model. This is the 8B-parameter dense version, focused here on the text-to-image capability.

## What makes it different

Most image generators today are diffusion models built around a separate variational autoencoder (VAE) that compresses pixels into a latent space. SenseNova U1 throws that out: there is no VAE and no separate vision encoder. Text and visual information are modelled end-to-end as a unified compound, using an architecture the SenseNova team calls NEO-Unify. The same model also natively understands images, edits them, and can interleave generated text with generated images in a single response.

In practice that means you get a model that is comfortable with dense, text-heavy, infographic-style images — not just artistic scenes — because the language and pixel paths share the same representation throughout.

## Strengths

- Photorealistic and stylized text-to-image generation at native ~2K resolutions.
- Strong text rendering inside images: posters, slides, infographics, comics, resumes.
- Faithful, structured layouts when prompts describe complex scenes with multiple elements.
- Reasoning-aware generation: the model can think about a prompt before drawing.

## How to use

Write your prompt the way you would describe the image to a designer. The more specific you are about subject, composition, lighting, and style, the closer the output will be to what you want. The model is trained on 2K resolution buckets — the dropdown lists every supported aspect ratio. Picking one outside the trained set is allowed but may hurt quality.

Useful knobs:

- **CFG scale** controls how strictly the model follows the prompt. Higher values track the prompt more closely; lower values allow more variety. The default of 4.0 is a good starting point.
- **Steps** controls the quality vs. speed tradeoff. 30-50 is a sweet spot. More steps rarely help past 50.
- **Seed** locks in reproducibility. Set it to a fixed integer if you want to iterate on a prompt without the image changing underneath you.

## Hardware and performance

This model runs on a single Nvidia L40S GPU. Cold boots take about two minutes; once warm, a 2048x2048 image at 30 steps takes about 30 seconds.

## License

The model weights and the SenseNova U1 codebase are released under Apache 2.0 by SenseNova. This Cog wrapper is also Apache 2.0. Read the SenseNova U1 model card on Hugging Face for the full upstream details and benchmark numbers.

## Credits

SenseNova U1 was created by the OpenSenseNova team. The original code, weights, and inference scripts are at huggingface.co/sensenova/SenseNova-U1-8B-MoT and github.com/OpenSenseNova/SenseNova-U1.
