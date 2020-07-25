import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
import mlflow
from mlflow import keras, onnx
import datetime
import click
import keras2onnx
import onnxruntime

@click.command(help="Description...")
@click.option("--train-data-csv", default="data/train.csv", help="path of input train csv file")
@click.option("--test-data-csv", default="data/test.csv", help="path of input test csv file")
@click.option("--batch-size", default=64) # default 32
@click.option("--epochs", default=1) # default 2
@click.option("--max-features", default=2000) #default 2000
@click.option("--max-len", default=200, help="max length of words of a comment - longer ones get trimmed")
@click.option("--embed-size", default=256, help="define the size of the vector space") #default = 128
@click.option("--model-name", default="model.onnx", help="name of the exported model")
@click.option("--model-path", default="model", help="path where to safe the exported model")
def train_model(train_data_csv, test_data_csv, batch_size, epochs, max_features, max_len, embed_size, model_name, model_path):
    with mlflow.start_run(run_name="Train Model") as mlrun:
        mlflow.set_tag("Start Time", datetime.datetime.now())

        # read Data from given CSV files
        print('Read Data from given CSV files ...')
        train = pd.read_csv(train_data_csv)
        test = pd.read_csv(test_data_csv)

        # log parameters for MLflow Tracking
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('max_features', max_features)
        mlflow.log_param('maxlen', max_len)
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

        totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

        # building the model
        # defining an Input layer that accepts a list of sentences that has a dimension of maxlen (default=200)
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

        # autolog metrics, parameters, and model
        keras.autolog()

        # fitting the model
        print("\nstarting to train model...")
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        # Evaluation of the Model against the Test datasets
        score = model.evaluate(X_test, y_test)
        print('\nTest loss:', score[0])
        mlflow.log_metric('Test Dataset loss', score[0])
        print('Test accuracy:', score[1])
        mlflow.log_metric('Test Dataset accuracy', score[1])

        # convert to serialized onnx model
        print("\nConvert model to ONNX format ...")
        onnx_model = keras2onnx.convert_keras(model, model.name)

        # create directory if not exists
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # save model and log to mlflow
        model_path = os.path.join(model_path, model_name)
        print("saveing ONNX model to %s", model_path)
        keras2onnx.save_model(onnx_model, model_path)
        mlflow.onnx.log_model(onnx_model, model_path)

        mlflow.set_tag("End Time", datetime.datetime.now())

if __name__ == '__main__':
    train_model()
