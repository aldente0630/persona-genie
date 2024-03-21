import ast
import base64
import os
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
from transformers import CLIPFeatureExtractor


class CkptName(str, Enum):
    IP_ADAPTER_FACEID_PLUS_V1: str = "ip-adapter-faceid-plus_sd15.bin"
    IP_ADAPTER_FACEID_PLUS_V2 = "ip-adapter-faceid-plusv2_sd15.bin"


class HfModelId(str, Enum):
    SD_V1_5: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    SD_VAE: str = "stabilityai/sd-vae-ft-mse"
    SD_SAFETY_CHECKER: str = "CompVis/stable-diffusion-safety-checker"
    CLIP_VIT_BASE_PATCH32: str = "openai/clip-vit-base-patch32"
    IMAGE_ENCODER: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def model_fn(model_dir: str) -> Dict[str, Any]:
    use_shortcut = ast.literal_eval(os.getenv("USE_SHORTCUT", "False"))

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(HfModelId.SD_VAE.value).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        HfModelId.SD_V1_5.value,
        scheduler=scheduler,
        vae=vae,
        torch_dtype=torch.float16,
        feature_extractor=CLIPFeatureExtractor.from_pretrained(
            HfModelId.CLIP_VIT_BASE_PATCH32.value,
        ),
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            HfModelId.SD_SAFETY_CHECKER.value,
        ),
    ).to(device)

    model = IPAdapterFaceIDPlus(
        pipe,
        HfModelId.IMAGE_ENCODER.value,
        os.path.join(
            model_dir,
            CkptName.IP_ADAPTER_FACEID_PLUS_V2.value
            if use_shortcut
            else CkptName.IP_ADAPTER_FACEID_PLUS_V1.value,
        ),
        device,
    )

    return {"app": app, "model": model}


def predict_fn(data: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, List[str]]:
    use_shortcut = ast.literal_eval(os.getenv("USE_SHORTCUT", "False"))
    app, model = models["app"], models["model"]

    prompt = data["prompt"]
    negative_prompt = data.pop("negative_prompt")
    s_scale = data.pop("s_scale", 1.0)
    num_samples = data.pop("num_samples", 4)
    width = data.pop("width", 512)
    height = data.pop("height", 768)
    num_inference_steps = data.pop("num_inference_steps", 30)
    seed = data.pop("seed", 42)

    if negative_prompt is not None and len(negative_prompt) > 0:
        negative_prompt = None

    image = np.array(
        Image.open(BytesIO(base64.b64decode(data["image"].encode()))).convert("RGB")
    )

    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

    generated_images = model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        face_image=face_image,
        faceid_embeds=faceid_embeds,
        shortcut=use_shortcut,
        s_scale=s_scale,
        num_samples=num_samples,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    return {"generated_images": encoded_images}