from typing import Dict
from kedro.pipeline import Pipeline
from kedro_pipeline.pipelines import ml_pipeline as ml


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:

    _ml_pipeline = ml.create_pipeline()

    return {
        "ml": _ml_pipeline,
        "__default__": _ml_pipeline
    }
