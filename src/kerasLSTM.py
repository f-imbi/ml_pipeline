import datetime
import json
import os
import pandas as pd
import click
import keras2onnx
import mlflow
import yaml
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from mlflow import keras, onnx
from sklearn import metrics


@click.command(help="Description...")
@click.option("--train-data-csv", default="data/train.csv", help="path of input train csv file")
@click.option("--test-data-csv", default="data/test.csv", help="path of input test csv file")
@click.option("--batch-size", default=32)  # default 32
@click.option("--epochs", default=1)  # default 2
@click.option("--max-features", default=2000)  # default 2000
@click.option("--max-len", default=200, help="max length of words of a comment - longer ones get trimmed")
@click.option("--embed-size", default=128, help="define the size of the vector space")  # default = 128
@click.option("--model-name", default="model.onnx", help="name of the exported model")
@click.option("--model-path", default="model", help="path where to safe the exported model")
def call_train_model(train_data_csv, test_data_csv, batch_size, epochs, max_features, max_len, embed_size, model_name, model_path):
    train_model(train_data_csv, test_data_csv, batch_size, epochs, max_features, max_len, embed_size, model_name, model_path)


def train_model(train_data, test_data, batch_size, epochs, max_features, max_len, embed_size, model_name,
                model_path):
    with mlflow.start_run(run_name="Train Model") as mlrun:
        mlflow.set_tag("Start Time", datetime.datetime.now())

        # read Data from given DataFrame or CSV file
        print('Read Data from given DataFrame or CSV file ...')
        if isinstance(train_data, str) & isinstance(test_data, str):
            train = pd.read_csv(train_data)
            test = pd.read_csv(test_data)
        if isinstance(train_data, pd.DataFrame) & isinstance(test_data, pd.DataFrame):
            train = train_data
            test = test_data

        # if params.yaml file exists, import params from it
        if os.path.isfile("params.yaml"):
            params = yaml.safe_load(open("params.yaml"))['train_keras']
            batch_size = params['batch_size']
            epochs = params['epochs']
            max_features = params['max_features']
            max_len = params['max_len']
            embed_size = params['embed_size']

        # log parameters for MLflow Tracking
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('max_features', max_features)
        mlflow.log_param('max_len', max_len)
        mlflow.log_param('embed_size', embed_size)

        # the dependent variables are in the training set itself so we need to split them up, into X and Y sets
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        y_train = train[list_classes].values
        y_test = test[list_classes].values
        list_sentences_train = train["comment_text"]
        list_sentences_test = test["comment_text"]

        # tokenizing
        print("tokenizing the sentences ...")
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(list_sentences_train))
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
        X_train = pad_sequences(list_tokenized_train, maxlen=max_len)
        X_test = pad_sequences(list_tokenized_test, maxlen=max_len)

        # building the model
        model = build_model(max_len, max_features, embed_size)

        # autolog metrics, parameters, and model
        keras.autolog()

        # fitting the model
        print("\nstarting to train model...")
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        # evaluation of the model
        model_metrics = evaluation(model, test, X_test, y_test, list_classes)

        # convert model to serialized onnx model and save it
        onnx_model = save_model(model, model_path, model_name)
        mlflow.set_tag("End Time", datetime.datetime.now())
        return onnx_model, model_metrics


def build_model(max_len, max_features, embed_size):
    # defining an Input layer that accepts a list of sentences that has a dimension of max_len (default=200)
    print("\nbulding the model with max_len: %s, max_features: %s, embed_size: %s"
          % (max_len, max_features, embed_size))
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size)(inp)
    x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("model summary:")
    model.summary()
    return model


def evaluation(model, test, X_test, y_test, categories):

    # Evaluation of the Model against the Test datasets
    print("\nmodel evaluation ...")
    score = model.evaluate(X_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # log loss and accuracy metrics to mlflow tracking
    mlflow.log_metric('Test loss', score[0])
    mlflow.log_metric('Test accuracy', score[1])

    # prediction of the Model against the Test datasets
    print("\nmodel prediction ...")
    pred = model.predict(X_test)
    pred = pd.DataFrame(pred)
    cols = [0, 1, 2, 3, 4, 5]

    # save prediction to csv file in metrics folder
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    pred.to_csv("metrics/prediction.csv", index=False)
    mlflow.log_artifact("metrics/prediction.csv")

    # calculate r2_score and mean_squared_error (Regression metrics)
    r2_score = metrics.r2_score(test[categories], pred[cols])
    mean_squared_error = metrics.mean_squared_error(test[categories], pred[cols])
    print("\nRegression metrics:")
    print("r2_score: " + str(r2_score))
    print("mean_squared_error: " + str(mean_squared_error))
    # log regression metrics to mlflow tracking
    mlflow.log_metric("r2_score", r2_score)
    mlflow.log_metric("mean_squared_error", mean_squared_error)

    # convert predictions to binary for calculation of classification metrics
    for col in cols:
        pred[col] = [1 if row > 0.5 else 0 for row in pred[col]]

    # calculate precision, recall, f1_score, roc_auc (Classification metrics)
    precision = metrics.precision_score(test[categories], pred[cols], average='weighted')
    recall = metrics.recall_score(test[categories], pred[cols], average='weighted')
    f1_score = metrics.f1_score(test[categories], pred[cols], average='weighted')
    roc_auc = metrics.roc_auc_score(test[categories], pred[cols])
    print("\nClassification metrics:")
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1_score: " + str(f1_score))
    print("roc_auc: " + str(roc_auc))
    # log classification metrics to mlflow tracking
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("roc_auc", roc_auc)

    # save all metrics in metrics.json file
    metrics_dump = {
        'evaluation_metrics': {
            'test_loss': score[0],
            'test_accuracy': score[1]
        },
        'regression_metrics': {
            'r2_score': r2_score,
            'mean_squared_error': mean_squared_error
        },
        'classification_metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc
        }
    }

    # save file and log as mlflow artifact
    with open('metrics/metrics.json', 'w') as fd:
        json.dump(metrics_dump, fd, indent=2)
    mlflow.log_artifact('metrics/metrics.json', 'metrics')
    return metrics_dump


def save_model(model, model_path, model_name):
    print("\nConvert model to ONNX format ...")
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # create directory if not exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # save model and log to mlflow
    model_path = os.path.join(model_path, model_name)
    print("saving ONNX model to ", model_path)
    keras2onnx.save_model(onnx_model, model_path)
    mlflow.onnx.log_model(onnx_model, model_path)
    return onnx_model


if __name__ == '__main__':
    call_train_model()
