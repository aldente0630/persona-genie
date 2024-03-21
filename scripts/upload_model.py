import os
import random
import shutil
import sys
import boto3
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.config_handler import load_config
from utils.enums import CkptName, DirName, FileName, Url
from utils.logger import logger
from utils.misc import compress_dir_to_model_tar_gz, get_dir_path


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    config_dir = get_dir_path(os.path.join(os.pardir, DirName.CONFIGS.value))
    config = load_config(os.path.join(config_dir, FileName.CONFIG.value))

    # Download the model checkpoint from the HuggingFace repository
    CKPT_NAME = (
        CkptName.IP_ADAPTER_FACEID_PLUS_V2.value
        if config.use_shortcut
        else CkptName.IP_ADAPTER_FACEID_PLUS_V1.value
    )
    response = requests.get(
        f"https://{Url.HF_IP_ADAPTER_FACEID.value}/{CKPT_NAME}?download=true"
    )

    models_dir = os.path.join(os.pardir, DirName.MODELS.value)
    os.makedirs(models_dir, exist_ok=True)
    models_path = os.path.join(models_dir, CKPT_NAME)
    status_code = response.status_code

    if status_code == 200:
        with open(models_path, "wb") as file:
            file.write(response.content)
        logger.info("File downloaded successfully and saved as '%s'.", models_path)

    else:
        raise requests.exceptions.HTTPError(
            f"Failed to download the file: {status_code}"
        )

    # Upload the model checkpoint and source code to the S3 bucket
    source_dir = os.path.join(os.pardir, f"source-{random.getrandbits(16)}")
    os.makedirs(source_dir, exist_ok=True)

    shutil.copyfile(models_path, os.path.join(source_dir, CKPT_NAME))
    shutil.copytree(
        os.path.join(os.pardir, DirName.CODE.value, DirName.SAGEMAKER.value)
        + os.path.sep,
        os.path.join(source_dir, "code"),
    )

    compress_dir_to_model_tar_gz(source_dir, output_file="model.tar.gz", logger=logger)
    shutil.rmtree(source_dir)

    boto_session = boto3.Session(
        region_name=config.region_name, profile_name=config.profile_name
    )
    s3_client = boto_session.client("s3")
    model_uri = f"{config.proj_name}/model.tar.gz"

    response = s3_client.upload_file("model.tar.gz", config.bucket_name, model_uri)
    os.remove("model.tar.gz")

    logger.info(
        "File uploaded successfully to '%s'.", f"{config.bucket_name}/{model_uri}"
    )
