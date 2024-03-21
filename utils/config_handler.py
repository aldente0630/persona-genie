import os
import sys
from dataclasses import dataclass
from typing import Optional
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.misc import get_default


@dataclass
class Config:
    proj_name: str
    region_name: str
    profile_name: Optional[str]
    bucket_name: str
    instance_type: str
    use_shortcut: bool


def load_config(config_path: str) -> Config:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    bucket_name = config["proj"]["s3_bucket_name"]
    assert (
        bucket_name is not None
    ), "Please provide your S3 bucket name in 'configs/config.yaml'."

    return Config(
        proj_name=get_default(config["proj"]["proj_name"], "persona-genie"),
        region_name=get_default(config["proj"]["region_name"], "us-east-1"),
        profile_name=config["proj"]["iam_profile_name"],
        bucket_name=bucket_name,
        instance_type=get_default(config["proj"]["sm_instance_type"], "ml.g4dn.xlarge"),
        use_shortcut=get_default(config["model"]["use_shortcut"], False),
    )
