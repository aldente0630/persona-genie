from enum import Enum


class CkptName(str, Enum):
    IP_ADAPTER_FACEID_PLUS_V1: str = "ip-adapter-faceid-plus_sd15.bin"
    IP_ADAPTER_FACEID_PLUS_V2 = "ip-adapter-faceid-plusv2_sd15.bin"


class DirName(str, Enum):
    ASSETS: str = "assets"
    CODE: str = "code"
    CONFIGS: str = "configs"
    LAMBDA: str = "lambda"
    LOGS: str = "logs"
    MODELS: str = "models"
    SAGEMAKER: str = "sagemaker"


class FileName(str, Enum):
    CONFIG: str = "config.yaml"
    LOG: str = "log.txt"


class Url(str, Enum):
    HF_IP_ADAPTER_FACEID: str = "huggingface.co/h94/IP-Adapter-FaceID/resolve/main"
    INFERENCE: str = "inference"
