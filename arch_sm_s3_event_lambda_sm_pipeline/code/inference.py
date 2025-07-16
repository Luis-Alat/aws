import joblib
import os

# Load Model
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

# Call Model
def predict_fn(input_data, model):
    return model.predict(input_data)

