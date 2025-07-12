import json
import logging
import pathlib
import joblib
import tarfile

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":

    logger.debug("Starting evaluation.")

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading RandomForest model.")
    
    model = joblib.load("model.joblib")

    logger.debug("Reading test data.")

    test_path = "/opt/ml/processing/data/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    
    y_test = df.iloc[:, 0].to_numpy()
    X_test = df.iloc[:, 1:].values

    logger.info("Performing predictions against test data.")

    predictions = model.predict(X_test)

    logger.debug("Calculating mean squared error.")

    f1  = f1_score(y_test, predictions, average="weighted")
    acc = accuracy_score(y_test, predictions)
    re  = recall_score(y_test, predictions, average="weighted")
    pre = precision_score(y_test, predictions, average="weighted")
    
    report_dict = {
        "classification_metrics": {
            "f1":  {"value": f1},
            "acc": {"value": acc},
            "re":  {"value": re},
            "pre": {"value": pre}
        }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report")

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))