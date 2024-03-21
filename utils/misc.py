import base64
from io import BytesIO
import logging
import os
import tarfile
from typing import Any, List, Optional
import matplotlib.pyplot as plt
from PIL import Image


def compress_dir_to_model_tar_gz(
    tar_dir: Optional[str] = None,
    output_file: str = "model.tar.gz",
    logger: Optional[logging.Logger] = None,
) -> None:
    parent_dir = os.getcwd()
    os.chdir(tar_dir)

    msg = "The following directories and files will be compressed."
    log_or_print(msg, logger=logger)

    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir("."):
            log_or_print(item, logger=logger)
            tar.add(item, arcname=item)

    os.chdir(parent_dir)


def decode_base64_image(image_string: str) -> Image:
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


def display_images(
    images: List[Image.Image],
    n_columns: int = 3,
    fig_size: int = 20,
) -> None:
    n_columns = min(len(images), n_columns)
    quotient, remainder = divmod(len(images), n_columns)
    if remainder > 0:
        quotient += 1
    width, height = images[0].size
    plt.figure(figsize=(fig_size, fig_size / n_columns * quotient * height / width))
    for i, image in enumerate(images):
        plt.subplot(quotient, n_columns, i + 1)
        plt.axis("off")
        plt.imshow(image, aspect="auto")
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()


def get_default(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


def log_or_print(msg: str, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.info(msg)
    else:
        print(msg)
