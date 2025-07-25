import argparse
import logging
from glob import glob
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder
)
from sklearn.compose import ColumnTransformer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":

    logger.debug("Starting preprocessing.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/input_data"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/opt/ml/processing"
    )
    args = parser.parse_args()

    input_data = args.input_data
    base_dir = args.base_dir

    logger.info(f"Loading data from: {input_data}")

    csv_files = glob(os.path.join(input_data, "*.csv"))

    if len(csv_files) == 1:
        df = pd.read_csv(csv_files[0])
    elif len(csv_files) > 1:
        df = pd.concat([pd.read_csv(file) for file in csv_files])
    else:
        raise ValueError("Zero csv files were found. Check at least one CSV is present in the respective folder") 

    df.drop("LoanID", inplace=True, axis=1)

    logger.debug("Defining transformers.")

    # Defining the variables and expected values 
    # to transform (ordinal, nominal and numeric variables)
    ordinal_var = {
        "Education": [["High School", "Bachelor's", "Master's", "PhD"]],
        "EmploymentType": [['Unemployed', 'Part-time', 'Self-employed', 'Full-time']]
    }
    nominal_var = ["MaritalStatus", "LoanPurpose", "HasMortgage", "HasDependents", "HasCoSigner"]
    numeric_var = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']


    # Defining encoding to ordinal variables
    ordinal_pipeline = []
    for var_name, var_labels in ordinal_var.items():
        encoder = OrdinalEncoder(
            categories=var_labels,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype="int16"
        )
        encoder_transformer = (var_name[:3].lower(), encoder, [var_name])
        ordinal_pipeline.append(encoder_transformer)

    # Defining encoding to nominal variables
    one_hot_encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False, dtype="int8"
    )

    # Defining encoding to transform numeric variables
    standarization = StandardScaler()


    # Pipeline (ordinal + nominal [One Hot Encoder] + numeric [Standarization])
    transformer_pipeline = ordinal_pipeline
    transformer_pipeline += [
        ("ohe", one_hot_encoder, nominal_var),
        ("standarization", standarization, numeric_var)
    ]

    transformer = ColumnTransformer(
        transformers=transformer_pipeline,
        remainder='passthrough',
        verbose_feature_names_out=True
    )

    logger.info("Applying transforms.")
    transformed_data = transformer.fit_transform(df)

    transformed_data = pd.DataFrame(
        transformed_data,
        columns=transformer.get_feature_names_out()
    )
    y = transformed_data.pop("remainder__Default").astype("int8")
    transformed_data = pd.concat([y, transformed_data], axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(transformed_data))

    transformed_data = transformed_data.sample(frac=1)

    train, validation, test = np.split(
        transformed_data,
        [int(0.7 * len(transformed_data)), int(0.85 * len(transformed_data))]
    )

    logger.info("Writing out datasets to %s.", base_dir)

    #transformed_data.to_parquet("../data/processed/processed_data.parquet", index=False)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)