import json
import logging
import os
import sys
from typing import Any, Optional
import boto3
from botocore.exceptions import ClientError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.misc import log_or_print


def get_ssm_parameter(
    boto_session: boto3.Session,
    parameter_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    ssm_client = boto_session.client("ssm")
    try:
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        return response["Parameter"]["Value"]

    except ClientError:
        log_or_print("The requested parameter name doesn't exist.", logger=logger)
        return None


def invoke_sagemaker_endpoint(
    boto_session: boto3.Session,
    endpoint_name: str,
    payload: Any,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    sm_runtime_client = boto_session.client("sagemaker-runtime")

    try:
        response = sm_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        return result

    except ClientError as error:
        log_or_print(str(error), logger=logger)
        return None
