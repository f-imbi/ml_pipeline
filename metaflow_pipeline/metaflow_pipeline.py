from metaflow import FlowSpec, step, Parameter, IncludeFile
import sys
sys.path.append("..")
from src.download_raw_data import load_raw_data
from src.validate_data import validate_data
from src.split_data import split_data
from src.kerasLSTM import train_model


class ML_Pipeline(FlowSpec):
    """
    A flow where Metaflow runs the steps of a ML Pipeline
    """
    url = Parameter('url', default='https://cloud.beuth-hochschule.de/index.php/s/P3g95HY68taz78g/download')
    # Params for train_test_split
    test_size = Parameter('test_size', default=0.2)
    random_state = Parameter('random_state', default=42)
    # Params for train_keras
    batch_size = Parameter('batch_size', default=32)
    epochs = Parameter('epochs', default=1)
    max_features = Parameter('max_features', default=2000)
    max_len = Parameter('max_len', default=200)
    embed_size = Parameter('embed_size', default=128)

    @step
    def start(self):
        """
        STEP 1 of ML Pipeline - Download Data
        """
        print("ML Pipeline is starting ...")
        self.raw_data_filepath = load_raw_data(self.url, "data", "raw_data.csv", "raw_data.zip")
        self.next(self.validation)

    @step
    def validation(self):
        """
        STEP 2 of ML Pipeline - Data Validation
        """
        print("STEP 2")
        self.validated_data_df = validate_data(self.raw_data_filepath, "data/validated_data.csv")
        self.next(self.split_data)

    @step
    def split_data(self):
        """
        STEP 3 of ML Pipeline - Split Data
        """
        print("STEP 3")
        self.train, self.test = split_data(self.validated_data_df, self.test_size,
                                           self.random_state, True, "data/train.csv",
                                           "data/test.csv")
        self.next(self.end)

    @step
    def end(self):
        """
        STEP 4 of ML Pipeline - Train & Evaluate Keras Model
        """
        print("STEP 4")
        self.model, self.metrics = train_model(self.train, self.test, self.batch_size,
                                               self.epochs, self.max_features, self.max_len,
                                               self.embed_size, "model.onnx", "model")
        print("ML Pipeline is all done.")


if __name__ == '__main__':
    ML_Pipeline()
