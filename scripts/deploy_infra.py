import os
import sys
from typing import Optional
import aws_cdk as core
from aws_cdk import (
    aws_apigateway as api_gateway,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_sagemaker as sagemaker,
    aws_ssm as ssm,
)
from constructs import Construct

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.config_handler import load_config
from utils.enums import DirName, FileName, Url
from utils.misc import get_dir_path


class ImageGenModelStack(core.Stack):
    def __init__(
        self,
        scope: Optional[Construct] = None,
        id: Optional[str] = None,
        proj_name: Optional[str] = None,
        instance_type: Optional[str] = None,
        model_data_url: Optional[str] = None,
        use_shortcut: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Create an IAM role for a SageMaker Endpoint
        sm_role = iam.Role(
            self,
            "Iam-ImageGen-SageMaker-Role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            role_name=f"{proj_name}-sagemaker-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
            ],
        )

        # Create an IAM role for a Lambda Function
        lambda_role = iam.Role(
            self,
            "Iam-ImageGen-Lambda-Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name=f"{proj_name}-lambda-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )

        model = sagemaker.CfnModel(
            self,
            "SageMaker-ImageGen-Model",
            execution_role_arn=sm_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                environment={"USE_SHORTCUT": str(use_shortcut)},
                image=get_hf_infer_container_url(
                    str(core.Aws.REGION), "2.1.0", use_gpu=True
                ),
                model_data_url=model_data_url,
            ),
        )

        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "SageMaker-ImageGen-EndpointConfig",
            production_variants=[
                {
                    "modelName": model.attr_model_name,
                    "initialInstanceCount": 1,
                    "instanceType": instance_type,
                    "variantName": "AllTraffic",
                }
            ],
            endpoint_config_name=f"{proj_name}-endpoint",
        )

        _ = sagemaker.CfnEndpoint(
            self,
            "SageMaker-ImageGen-Endpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
            endpoint_name=f"{proj_name}-endpoint",
        )

        # Create the Lambda Function
        function = lambda_.Function(
            self,
            "Lambda-ImageGen-Function",
            code=lambda_.Code.from_asset(os.path.join(DirName.CODE, DirName.LAMBDA)),
            environment={
                "ENDPOINT_NAME": f"{proj_name}-endpoint",
            },
            function_name=f"{proj_name}-function",
            handler="app.handler",
            role=lambda_role,
            runtime=lambda_.Runtime.PYTHON_3_10,
            timeout=core.Duration.seconds(60),
        )

        # Create an API Gateway to expose the Lambda Function
        rest_api = api_gateway.LambdaRestApi(
            self,
            "ApiGateway-ImageGen-RestApi",
            handler=function,
            proxy=False,
            rest_api_name=f"{proj_name}-rest-api",
        )

        inference = rest_api.root.add_resource(Url.INFERENCE.value)
        inference.add_method("POST")

        # Store the URL of the API gateway in SSM Parameter Store
        ssm.StringParameter(
            self,
            "Ssm-ImageGen-Parameter",
            parameter_name=f"/{proj_name}/url",
            string_value=rest_api.url,
        )


def get_hf_infer_container_url(
    region_name: str,
    pytorch_version: str,
    use_gpu: bool = False,
) -> str:
    hf_infer_container_url_dict = (
        {
            "2.1.0": f"763104351884.dkr.ecr.{region_name}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04",
        }
        if use_gpu
        else {
            "2.1.0": f"763104351884.dkr.ecr.{region_name}.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04",
        }
    )

    return hf_infer_container_url_dict[pytorch_version]


if __name__ == "__main__":
    config_dir = get_dir_path(os.path.join(os.pardir, DirName.CONFIGS.value))
    config = load_config(os.path.join(config_dir, FileName.CONFIG.value))
    model_data_url = f"s3://{config.bucket_name}/{config.proj_name}/model.tar.gz"

    app = core.App()
    ImageGenModelStack(
        app,
        "ImageGenModelStack",
        proj_name=config.proj_name,
        instance_type=config.instance_type,
        model_data_url=model_data_url,
        use_shortcut=config.use_shortcut,
        env=core.Environment(region=config.region_name),
    )
    app.synth()
