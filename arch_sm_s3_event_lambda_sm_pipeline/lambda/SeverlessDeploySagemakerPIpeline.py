import json
import boto3
import time
import uuid

sagemaker = boto3.client('sagemaker')

def lambda_handler(event, context):

    model_package_arn = event['model_package_arn']

    unique_id = str(uuid.uuid4())[:8]
    model_name = f"deployed-model-{unique_id}"
    endpoint_config_name = f"serverless-config-{unique_id}"
    endpoint_name = f"serverless-endpoint-{unique_id}"

    create_model_response = sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "ModelPackageName": model_package_arn
        },
        ExecutionRoleArn="arn:aws:iam::007863746889:role/ModelArtefacts"
    )

    create_endpoint_config_response = sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'ServerlessConfig': {
                    'MemorySizeInMB': 2048,
                    'MaxConcurrency': 5
                }
            }
        ]
    )

    create_endpoint_response = sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    return {
        'statusCode': 200,
        'body': json.dumps({
            'EndpointName': endpoint_name,
            'ModelName': model_name,
            'EndpointConfigName': endpoint_config_name
        })
    }

