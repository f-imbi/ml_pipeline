import requests
import os
import zipfile
import mlflow
import datetime
import click


@click.command(help="Downloads a ZIP packed data set, extraxts it and saves it as an mlflow artifact")
@click.option("--url", default="https://cloud.beuth-hochschule.de/index.php/s/P3g95HY68taz78g/download")
@click.option("--path", default="data", help="Path where to store the dataset")
@click.option("--filename", default="raw_data.csv", help="name given to the downloaded and extracted dataset")
@click.option("--zip-filename", default="raw_data.zip", help="name of the zipped dataset downloaded from source (url)")
def call_load_raw_data(url, path, filename, zip_filename):
    load_raw_data(url, path, filename, zip_filename)


def load_raw_data(url, path, filename, zip_filename):
    with mlflow.start_run(run_name="Download Data") as mlrun:
        mlflow.set_tag("Start Time", datetime.datetime.now())
        
        # create directory if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        # download file if not exists
        if not os.path.exists(os.path.join(path, zip_filename)):
            zip_filepath = os.path.join(path, zip_filename)
            print("Downloading %s to %s" % (url, zip_filepath))
            mlflow.set_tag("Start Time Download", datetime.datetime.now())
            r = requests.get(url, stream=True)
            mlflow.set_tag("End Time Download", datetime.datetime.now())

            with open(zip_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

            # extract zip file
            print("Extracting %s into %s" % (zip_filepath, path))
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(path)

            # rename file to given filename
            for files in os.listdir(path):
                if files.endswith(".csv"):
                    print("Rename extracted file %s into %s" % (files, filename))
                    os.rename(os.path.join(path, files), os.path.join(path, filename))

        # save as mlflow artifact
        raw_data = os.path.join(path, filename)
        print("Save MLflow artifact: %s" % raw_data)
        mlflow.log_artifact(raw_data, "raw_data-csv-dir")
        mlflow.set_tag("End Time", datetime.datetime.now())
        return raw_data


if __name__ == '__main__':
    call_load_raw_data()
