import base64
import io
import os
import sys
from typing import Final, List, Optional
import boto3
import numpy as np
import requests
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.aws_utils import get_ssm_parameter
from utils.config_handler import load_config
from utils.enums import DirName, FileName, Url
from utils.logger import logger
from utils.misc import decode_base64_image, get_default, get_dir_path

NEGATIVE_PROMPT: Final = """
cleavage, nipples, nsfw, nude, uncensored, bad anatomy, bad proportions, bad quality, blurry, collage, cropped, 
deformed, dehydrated, disconnected limbs, disfigured, disgusting, error, extra arms, extra hands, extra limbs, 
fused fingers, grainy, gross proportions, jpeg, jpeg artifacts, long neck, low quality, low res, malformed limbs, 
missing arms, missing fingers, mutated, mutated hands, mutated limbs, out of frame, out of focus, picture frame, 
pixel, pixelated, poorly drawn face, poorly drawn hands, signature, text, ugly, username, watermark, worst quality 
"""


class App:
    def __init__(
        self,
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        proj_name: Optional[str] = None,
    ) -> None:
        self.region_name = region_name
        self.profile_name = profile_name
        self.proj_name = proj_name

        boto_session = boto3.Session(
            region_name=self.region_name, profile_name=self.profile_name
        )
        self.url = get_ssm_parameter(boto_session, f"/{self.proj_name}/url")

    def invoke(
        self,
        image: np.array,
        prompt: str,
        negative_prompt: Optional[str] = None,
        s_scale: float = 1.0,
        num_samples: int = 2,
        width: int = 512,
        height: int = 768,
        num_inference_steps: int = 30,
        seed: int = 42,
    ) -> List[np.array]:
        try:
            bytes_io = io.BytesIO()
            Image.fromarray(image).save(bytes_io, format="JPEG")

            payload = {
                "prompt": prompt,
                "negative_prompt": get_default(negative_prompt, NEGATIVE_PROMPT),
                "image": base64.b64encode(bytearray(bytes_io.getvalue())).decode(),
                "s_scale": s_scale,
                "num_samples": num_samples,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
            }

            response = requests.post(f"{self.url}/{Url.INFERENCE.value}", json=payload)
            status_code = response.status_code

            if status_code == 200:
                return [
                    np.array(decode_base64_image(image))
                    for image in response.json()["generated_images"]
                ]

            logger.error("Failed to receive a valid response: %s", status_code)

        except AttributeError:
            logger.error("Received invalid inputs.")


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, DirName.CONFIGS.value))
    config = load_config(os.path.join(config_dir, FileName.CONFIG.value))

    app = App(
        region_name=config.region_name,
        profile_name=config.profile_name,
        proj_name=config.proj_name,
    )

    IMAGE_FILENAME = "input.jpg"  # Your sample image

    image_path = os.path.join(os.pardir, DirName.ASSETS, IMAGE_FILENAME)
    image = Image.open(image_path)

    PROMPT = "a photo of a man as the ironman"  # Your sample prompt

    images = app.invoke(image, PROMPT)

    if images is not None:
        for i, image in enumerate(images):
            image_path = os.path.join(os.pardir, DirName.ASSETS, f"output-{i}.jpg")
            image.save(image_path)
