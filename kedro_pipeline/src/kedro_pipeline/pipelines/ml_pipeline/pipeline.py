from kedro.pipeline import Pipeline, node
from .nodes import _download_data, _validate_data, _split_data, _train_keras


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                _download_data,
                inputs="params:url",
                outputs="raw_data_filepath"
            ),
            node(
                _validate_data,
                inputs="raw_data_filepath",
                outputs="validated_data"
            ),
            node(
                _split_data,
                inputs=["validated_data", "params:test_size", "params:random_state"],
                outputs=["train", "test"]
            ),
            node(
                _train_keras,
                inputs=["train", "test", "params:batch_size", "params:epochs", "params:max_features",
                        "params:max_len", "params:embed_size"],
                outputs=["model", "metrics"]
            )
        ]
    )
