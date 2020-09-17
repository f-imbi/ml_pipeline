# ML Pipeline Tool Evaluation

## Steps of the Pipeline
The Python scripts of the individual processing steps of the pipeline are located in the src folder.

* Step 1: The script download_raw_data.py is executed. The data set is downloaded as a file in ZIP format, unzipped and the CSV file is saved. Unless otherwise specified in the execution command, the data set is downloaded from the cloud storage of Beuth Cloud, a cloud solution of Beuth University of Applied Sciences based on Nextcloud.
    * small Dataset: https://cloud.beuth-hochschule.de/index.php/s/MBfP8MnRY395J2F/download
    * large Dataset: https://cloud.beuth-hochschule.de/index.php/s/P3g95HY68taz78g/download
* Step 2: validate_data.py is executed. The data set downloaded in step 1 is read in as a DataFrame using the Pandas Library. Using the Python library PandasSchema, the data set is now checked. First, the expected schema and valid values are defined and then validated against the imported data set. Missing values in the data set are filled with a placeholder. The validated and, if necessary, modified data set is again saved as a CSV file.
* Step 3: In this step, the execution of the script split_data.py ensures that the data set is divided into a training and a test data set. The data set is first read with Pandas and then split with the train_test_split() method from the Scikit-learn library. The execution command can be used to pass parameters for the ratio of the split and the random control when mixing the data.
* Step 4: Finally, by running kerasLSTM.py, the configuration, training and evaluation of a machine learning model is done. The model is then configured using LSTM and Word Embeddings. The training of the model is followed by an evaluation. The model makes predictions for the data from the test data set. From the prediction results, metrics for the quality of the model are calculated using the Scikit-learn library. The metrics are then written and stored in a file in JSON format. Finally, the evaluated model is converted into a serialized object in ONNX format and stored, which is realized with the help ofkeras2onnx.

## Pipeline Definition Files 
The files of each tool in which the pipeline is defined

### MLflow
* Definition of pipeline & params: mlflow_pipeline/MLproject & src/main.py

### DVC
* Definition of pipeline: dvc_pipeline/dvc.yaml
* Definition of params: dvc_pipeline/params.yaml

### Metaflow
* Definition of pipeline & params: metaflow_pipeline/metaflow_pipeline.py

### Kedro
* Definition of pipeline: 
    * kedro_pipeline/src/kedro_pipeline/pipelines/ml_pipeline/nodes.py
    * kedro_pipeline/src/kedro_pipeline/pipelines/ml_pipeline/pipeline.py
    * kedro_pipeline/src/kedro_pipeline/pipeline.py
* Definition of params: kedro_pipeline/conf/base/parameters.yml
