import json
import os
from typing import Any, Dict
import boto3
from botocore.exceptions import ClientError

sm_runtime_client = boto3.client("sagemaker-runtime")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    endpoint_name = os.environ["ENDPOINT_NAME"]

    try:
        response = sm_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=event["body"],
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": response["Body"].read().decode("utf-8"),
        }

    except ClientError as error:
        return {"statusCode": 500, "body": json.dumps({"error": str(error)})}
