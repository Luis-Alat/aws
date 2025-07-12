from pathlib import Path
import os


def get_current_folder(global_varibles):
    
    if "__file__" in global_varibles:
        current_file = Path(global_varibles["__file__"])
        current_folder = current_file.parent.resolve()
    else:
        current_folder = Path(os.getcwd())
    
    return current_folder

current_folder = get_current_folder(globals())
solution_prefix = "sagemaker-soln-documents-"

tag_key = "sagemaker-soln-documents-"
training_instance_type = "ml.c4.2xlarge"
inference_instance_type = "ml.c4.2xlarge"
hosting_instance_type = "ml.c4.2xlarge"
my_aws_role = ""
bucket_name = ""
model_id = "tensorflow-tc-bert-en-uncased-L-12-H-768-A-12-2"