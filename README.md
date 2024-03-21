# 🧞‍♂️ Persona Genie
### Create your own AI avatar photo in 10 seconds!
- - -
![AI Avatar](./assets/avatar.png)  
![Demo](https://github.com/aldente0630/persona-genie/assets/10204855/1ba5b405-6b19-4d0b-bbca-81eb936fd429)

This application leverages the [IP-Adapter](https://arxiv.org/abs/2308.06721) model for AI avatar photo generation, developed with AWS services and [Gradio](https://www.gradio.app/) for an interactive interface. It utilizes the [AWS Cloud Development Kit](https://aws.amazon.com/cdk/) for quick infrastructure provisioning, making the setup process straightforward for demonstration purposes.

## 🛠 Usage Instructions
- - -
### Step 1: Preliminaries

- **Python Installation:** Ensure Python is installed in your environment.
- **AWS Configuration:** Set up your AWS credentials and configuration prior to deployment. This includes having access to an AWS account and the AWS CLI configured on your machine.
- **S3 Bucket Preparation:** If you do not already have an S3 bucket, create one. Update the `s3_bucket_name` in the `configs/config.yaml` file with your bucket's name. Make sure the bucket is in the same region as specified in your config file.

### Step 2: Install Required Packages

1. **Install the AWS CDK:**
   - Run `npm install -g aws-cdk` to install the AWS Cloud Development Kit globally on your machine.

2. **Install Python Dependencies:**
   - Execute `pip install -r requirements.txt` to install necessary Python packages for the app.

### Step 3: Provision Infrastructure

1. **Model Preparation:**
   - Download your model file to the HuggingFace repository and upload it to your S3 bucket using `python scripts/upload_model.py`.

2. **Deploy with CDK:**
   - Utilize the CDK to deploy SageMaker models, endpoints, Lambda functions, API Gateway, and more with `cdk deploy`.

### Step 4: Running Gradio

- Start the Gradio web application by running `python run.py`. Open the output URL in your browser to interact with the application.

### Step 5: Cleanup

- To avoid incurring unnecessary charges, stop the application by pressing `CTRL + C` and delete the deployed infrastructure with `cdk destroy`.

## 📝 Notes
- - -
- This guide assumes basic familiarity with AWS services and command line operations.
- For detailed documentation on each component, please refer to the respective AWS and Gradio documentation.

## 🤝 Contributions
- - -
- Contributions are welcome! Feel free to open issues or pull requests to improve the application or documentation.