import os
import click

import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id

# detection of if a run with the given entrypoint name,
# parameters, and experiment id already ran. The run must have completed
# successfully and have at least the parameters provided.
def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run_info.run_id, run_info.status))
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(("Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)") % (previous_version, git_commit))
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None

def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

@click.command()
@click.option("--url", default="https://cloud.beuth-hochschule.de/index.php/s/P3g95HY68taz78g/download")
@click.option("--test-size", default=0.2)
@click.option("--random-state", default=42)
@click.option("--batch-size", default=32)  # default 32
@click.option("--epochs", default=1)  # default 2
@click.option("--max-features", default=2000)  # default 2000
@click.option("--max-len", default=200, help="max length of words of a comment - longer ones get trimmed")
@click.option("--embed-size", default=128, help="define the size of the vector space")  # default = 128
def workflow(url, test_size, random_state, batch_size, epochs, max_features, max_len, embed_size):
    # The entrypoint names are defined in MLproject file. The artifact directories
    # are documented by each step's .py file
    with mlflow.start_run(run_name="ML Pipeline") as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        # step 1 download_data
        load_data = _get_or_run("download_data", {"url": url}, git_commit)
        raw_data_csv_uri = os.path.join(load_data.info.artifact_uri, "raw_data-csv-dir", "raw_data.csv")
        # step 2 validate_data with output of step 1
        validate_data = _get_or_run("validate_data", {"raw_data_csv": raw_data_csv_uri}, git_commit)
        validated_data_csv_uri = os.path.join(validate_data.info.artifact_uri, "validated_data-csv-dir",
                                              "validated_data.csv")
        # step 3 split_data with output of step 2
        split_data = _get_or_run("split_data",
                                 {"validated_data_csv": validated_data_csv_uri,
                                  "test_size": test_size,
                                  "random_state": random_state},
                                 git_commit)
        train_data_csv_uri = os.path.join(split_data.info.artifact_uri, "train-csv-dir", "train.csv")
        test_data_csv_uri = os.path.join(split_data.info.artifact_uri, "test-csv-dir", "test.csv")
        # step 4 train_keras with output of step 3
        train_keras = _get_or_run("train_keras",
                                  {"train_data_csv": train_data_csv_uri,
                                   "test_data_csv": test_data_csv_uri,
                                   "batch_size": batch_size,
                                   "epochs": epochs,
                                   "max_features": max_features,
                                   "max_len": max_len,
                                   "embed_size": embed_size},
                                  git_commit)

if __name__ == '__main__':
    workflow()