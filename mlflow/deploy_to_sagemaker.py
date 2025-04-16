from mlflow.deployments import get_deploy_client

region = 'eu-central-1'
# To get aws_id, aws support cli like "aws sts get-caller-identity"
aws_id = ''
# Use ARN from the role created in AWS with the full permission to Sagemaker
arn = f'arn:aws:iam::{aws_id}:role/time-series-project-role'
#app_name = ''
# find model uri in "mlflow ui" recorded as "logged_model"
run_id = ''
model_uri = f'runs:/{run_id}/trained_lstm'
# tag_id is from Docker image in AWS ECR
tag_id = '2.21.3'


image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id

config = dict(
    execution_role_arn=arn,
    #bucket_name="time-series-forecasting-project-data",
    image_url=image_url,
    region_name=region,
    archive=False,
    instance_type="ml.t3.medium",
    instance_count=1,
    synchronous=True,
    timeout_seconds=3600,
    variant_name="prod-variant-1",
    tags={"training_timestamp": "2025-04-14"},
)

client = get_deploy_client("sagemaker")

client.create_deployment(
    # app name displayed in Sagemaker
    "lstm-time-series-mlflow-deploy",
    model_uri=model_uri,
    flavor="python_function",
    config=config,
)


