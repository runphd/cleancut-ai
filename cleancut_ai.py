import time
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


MODEL_CACHE = Path("/tmp/stable_diffusion_model")
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

_pipelines = {}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sd_pipeline(pipeline_class):
    key = pipeline_class.__name__
    if key not in _pipelines:
        os.makedirs(MODEL_CACHE, exist_ok=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _pipelines[key] = pipeline_class.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            cache_dir=str(MODEL_CACHE),
            local_files_only=(MODEL_CACHE / "snapshots").exists(),
        ).to(_device())
    return _pipelines[key]


# ---------------------------------------------------------------------------
# Step 1 — Remove background
# ---------------------------------------------------------------------------

def remove_background(input_path: str, output_path: str) -> str:
    print(f"Removing background from {input_path}...")
    start = time.perf_counter()

    image = Image.open(input_path).convert("RGB")
    h, w = image.size[1], image.size[0]
    image_np = np.array(image)

    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4", trust_remote_code=True
    ).eval().to(_device())

    tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor = normalize(tensor / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    tensor = F.interpolate(tensor, size=(1024, 1024), mode="bilinear", align_corners=False).to(_device())

    with torch.no_grad():
        result = model(tensor)

    mask = F.interpolate(result[0][0], size=(h, w), mode="bilinear", align_corners=False).squeeze()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask_np = (mask * 255).cpu().numpy().astype(np.uint8)

    image_rgba = image.convert("RGBA")
    image_rgba.putalpha(Image.fromarray(mask_np, mode="L"))
    image_rgba.save(output_path, format="PNG")

    print(f"Done in {time.perf_counter() - start:.2f}s → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Step 2 — Generate a professional background suited to the object
# ---------------------------------------------------------------------------

def generate_background(
    object_desc: str,
    output_path: str,
    width: int = 768,
    height: int = 512,
) -> str:
    prompt = (
        f"professional product photography background for a {object_desc}, "
        f"studio lighting, clean composition, high quality, commercial photography, "
        f"bokeh, neutral tones, no people, no text"
    )
    negative_prompt = (
        f"{object_desc}, blurry, distorted, low quality, text, watermark, cartoon, anime"
    )

    print(f"Generating background for {object_desc}...")
    pipeline = _sd_pipeline(StableDiffusionPipeline)
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=40,
        guidance_scale=8.0,
    ).images[0]
    image.save(output_path)
    print(f"Background saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Step 3 — Composite object onto background
# ---------------------------------------------------------------------------

def composite_object(
    background_path: str,
    object_rgba_path: str,
    output_path: str,
) -> str:
    background = Image.open(background_path).convert("RGBA")
    obj = Image.open(object_rgba_path).convert("RGBA")

    bg_w, bg_h = background.size

    # Scale object to ~60% of background height, preserve aspect ratio
    target_h = int(bg_h * 0.60)
    scale = target_h / obj.size[1]
    target_w = int(obj.size[0] * scale)
    obj_resized = obj.resize((target_w, target_h), Image.LANCZOS)

    # Center the object
    paste_x = (bg_w - target_w) // 2
    paste_y = (bg_h - target_h) // 2

    composite = background.copy()
    composite.paste(obj_resized, (paste_x, paste_y), obj_resized)
    composite.convert("RGB").save(output_path)
    print(f"Composite saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Step 4 — Harmonize lighting and style with img2img
# ---------------------------------------------------------------------------

def harmonize(
    composite_path: str,
    object_desc: str,
    output_path: str,
) -> str:
    prompt = (
        f"professional product photo of a {object_desc}, studio lighting, "
        f"commercial photography, sharp focus, high quality"
    )
    negative_prompt = "blurry, distorted, low quality, text, watermark, cartoon, anime"

    print("Harmonizing composite...")
    composite = Image.open(composite_path).convert("RGB")
    pipeline = _sd_pipeline(StableDiffusionImg2ImgPipeline)
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=composite,
        strength=0.35,
        guidance_scale=8.0,
        num_inference_steps=40,
    ).images[0]
    result.save(output_path)
    print(f"Final image saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_image_path: str, object_desc: str) -> str:
    stem = Path(input_image_path).stem
    nobg_path      = f"{stem}_nobg.png"
    bg_path        = f"{stem}_background.png"
    composite_path = f"{stem}_composite.png"
    final_path     = f"{stem}_final.png"

    remove_background(input_image_path, nobg_path)
    generate_background(object_desc, bg_path)
    composite_object(bg_path, nobg_path, composite_path)
    harmonize(composite_path, object_desc, final_path)

    print(f"\nPipeline complete. Final image: {final_path}")
    return final_path


if __name__ == "__main__":
    run_pipeline("backpack.png", object_desc="backpack")
