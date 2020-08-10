import pandas as pd
import datetime
import mlflow
import click
import yaml
from sklearn import model_selection
from os import path

@click.command(
    help="Splits a given CSV file and logs the given train_test_split parameters."
         "The train and test files are saved as mlflow artifacts called 'train-data-csv' and 'test-data-csv'")
@click.option("--validated-data-csv", default="data/validated_data.csv", help="path of input csv file")
@click.option("--test-size", default=0.2)
@click.option("--random-state", default=42)
@click.option("--shuffle", default=True)
@click.option("--train-csv", default="data/train.csv", help="path of output train csv file")
@click.option("--test-csv", default="data/test.csv", help="path of output test csv file")
def call_split_data(validated_data_csv, test_size, random_state, shuffle, train_csv, test_csv):
    split_data(validated_data_csv, test_size, random_state, shuffle, train_csv, test_csv)


def split_data(validated_data, test_size, random_state, shuffle, train_csv, test_csv):
    with mlflow.start_run(run_name="Split Data") as mlrun:
        mlflow.set_tag("Start Time", datetime.datetime.now())

        # if params.yaml file exists, import params from it
        if path.isfile("params.yaml"):
            params = yaml.safe_load(open("params.yaml"))['split_data']
            test_size = params['test_size']
            random_state = params['random_state']

        # read Data from given DataFrame or CSV file
        print('Read Data from given DataFrame or CSV file')
        if isinstance(validated_data, str):
            df = pd.read_csv(validated_data)
        if isinstance(validated_data, pd.DataFrame):
            df = validated_data

        # split Dataframe to train and test with given parameters
        print('Split given CSV file with test_size=%s , random_state=%s and shuffle=%s'
              % (test_size, random_state, shuffle))
        train, test = model_selection.train_test_split(df, test_size=test_size, random_state=random_state,
                                                       shuffle=shuffle)

        # log train_test_split parameters to mlflow tracking UI
        print("log train_test_split parameters to mlflow tracking")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("train_size", (1 - test_size))
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("shuffle", shuffle)

        # determine the path where to save the train and test file
        train_path = path.join(train_csv)
        test_path = path.join(test_csv)

        # save the train and test file
        print("Save train and test csv files to %s and %s" % (train_path, test_path))
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        # save train & test csv files as mlflow artifacts
        print("Save MLflow artifacts: %s & %s" % (train_path, test_path))
        mlflow.log_artifact(train_path, "train-csv-dir")
        mlflow.log_artifact(test_path, "test-csv-dir")
        mlflow.set_tag("End Time", datetime.datetime.now())
        return train, test


if __name__ == '__main__':
    call_split_data()