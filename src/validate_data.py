import pandas as pd
import datetime
import mlflow
import click
import numpy as np
from pandas_schema import Schema, Column
from pandas_schema.validation import InListValidation, IsDtypeValidation
import os


@click.command(
    help="Validates and preprocesses a given CSV file and logs it's metrics."
         "The file is then saved as an mlflow artifact called 'validated-data-csv'")
@click.option("--raw-data-csv", default="data/raw_data.csv", help="path of input csv file")
@click.option("--validated-data-csv", default="data/validated_data.csv", help="path of validated output csv file")
def validate_data(raw_data_csv, validated_data_csv):
    with mlflow.start_run(run_name="Validate Data") as active_run:


        # Read Data from given CSV file
        print('Read Data from given CSV file')
        raw_data = pd.read_csv(raw_data_csv)

        # In case of missing comments, filling NA with "unknown"
        raw_data["comment_text"].fillna("unknown", inplace=True)

        # Define expected schema, data types and column values
        schema = defSchema()

        # Validate raw_data against expected Schema and column values
        print('\nValidate raw_data against expected Schema')
        errors = schema.validate(raw_data)

        # If errors, log them for MLflow Tracking UI
        if errors:
            x = 0
            for error in errors:
                x = x + 1
                print('Data Error:' + str(error))
                mlflow.set_tag(str(x) + '. Data Error', error)
                print('Run failed, please check errors')
            exit(1)
        print("Schema has been successfully validated")

        # log metrics to MLflow Tracking
        logMetrics(raw_data)

        # Save modified & validated data to csv file and save as mlflow artifact
        validated_data = os.path.join(validated_data_csv)
        raw_data.to_csv(path_or_buf=validated_data, index=False)
        mlflow.log_artifact(validated_data, "validated_data-csv-dir")
        mlflow.set_tag("End Time", datetime.datetime.now())


def defSchema():
    print('Define expected Schema')
    schema = Schema([
        Column(name='id', validations=[IsDtypeValidation(np.object_)], allow_empty=False),
        Column(name='comment_text', validations=[IsDtypeValidation(np.object_)], allow_empty=False),
        Column(name='toxic', validations=[InListValidation([0, 1])], allow_empty=False),
        Column(name='severe_toxic', validations=[InListValidation([0, 1])], allow_empty=False),
        Column(name='obscene', validations=[InListValidation([0, 1])], allow_empty=False),
        Column(name='threat', validations=[InListValidation([0, 1])], allow_empty=False),
        Column(name='insult', validations=[InListValidation([0, 1])], allow_empty=False),
        Column(name='identity_hate', validations=[InListValidation([0, 1])], allow_empty=False)
    ])
    return schema


def logMetrics(raw_data):
    print('\nLog Metrics to MLflow Tracking UI: ')
    # marking comments without any tags as "clean"
    x = raw_data.iloc[:, 2:].sum()
    rowsums = raw_data.iloc[:, 2:].sum(axis=1)
    raw_data['clean'] = (rowsums == 0)
    # count number of total comments
    print("\nTotal comments = ", len(raw_data))
    mlflow.log_metric("Total comments", len(raw_data))
    # count number of total tags
    print("Total tags =", x.sum())
    mlflow.log_metric("Total tags", x.sum())
    # count number of comments of each type and
    # log mean and standard deviation of each type for MLflow Tracking UI
    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'clean']
    for col in toxicity_cols:
        print("\ntotal " + col + " comments = ", raw_data[col].sum())
        mlflow.log_metric("total " + col + " comments", raw_data[col].sum())
        print(col + " mean = ", raw_data[col].mean())
        mlflow.log_metric(col + " mean", raw_data[col].mean())
        print(col + " std = ", raw_data[col].std())
        mlflow.log_metric(col + " std", raw_data[col].std())


if __name__ == '__main__':
    validate_data()
