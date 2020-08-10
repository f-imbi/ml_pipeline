from typing import Any, Dict
import pandas as pd
import sys
sys.path.append("..")
from src.download_raw_data import load_raw_data
from src.validate_data import validate_data
from src.split_data import split_data
from src.kerasLSTM import train_model


def _download_data(url: str):
    raw_data_filepath = load_raw_data(url, "data", "raw_data.csv", "raw_data.zip")
    return raw_data_filepath

def _validate_data(raw_data_filepath: str):
    validated_data = validate_data(raw_data_filepath, "data/validated_data.csv")
    return validated_data

def _split_data(validated_data: pd.DataFrame, test_size: float, random_state: int):
    train, test = split_data(validated_data, test_size, random_state, True,
                             "data/train.csv", "data/test.csv")
    return train, test

def _train_keras(train: pd.DataFrame, test: pd.DataFrame, batch_size: int, epochs: int,
                 max_features: int, max_len: int, embed_size: int):
    model, metrics = train_model(train, test, batch_size, epochs, max_features,
                                 max_len, embed_size, "model.onnx", "model")
    return model, metrics

