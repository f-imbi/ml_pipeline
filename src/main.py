import mlflow

def run(entrypoint):
    print("Launching new run for entrypoint=%s" % (entrypoint))
    submitted_run = mlflow.run(".", entry_point=entrypoint)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

def workflow():
    # The entrypoint names are defined in MLproject file. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run(run_name="ML Pipeline") as active_run:
        load_data = run("download_data")
        validate_data = run("validate_data")
        split_data = run("split_data")
        train_keras = run("train_keras")


if __name__ == '__main__':
    workflow()