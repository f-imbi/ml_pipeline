import pandas as pd
from sklearn import metrics
import json

def pred():

    # read Data from given CSV files
    print('Read Data from given CSV files ...')
    pred = pd.read_csv("prediction.csv")
    test = pd.read_csv("test.csv")
    categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    cols = ['0', '1', '2', '3', '4', '5']

    # calculate r2_score and mean_squared_error (Regression metrics)
    r2_score = metrics.r2_score(test[categories], pred[cols])
    mean_squared_error = metrics.mean_squared_error(test[categories], pred[cols])
    print("\nRegression metrics:")
    print("r2_score: " + str(r2_score))
    print("mean_squared_error: " + str(mean_squared_error))

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

    # save metrics in evaluation jason file like in evaluate.py (/home/fabian/src)
    metrics_dump = {}
    metrics_dump['evaluation_metrics'] = []
    metrics_dump['evaluation_metrics'].append({
        'r2_score': r2_score,
        'mean_squared_error': mean_squared_error
    })
    metrics_dump['regression_metrics'] = []
    metrics_dump['regression_metrics'].append({
        'r2_score': r2_score,
        'mean_squared_error': mean_squared_error
    })
    metrics_dump['classification_metrics'] = []
    metrics_dump['classification_metrics'].append({
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc
    })

    with open('metrics.json', 'w') as fd:
        json.dump(metrics_dump, fd, indent=4)


if __name__ == '__main__':
    pred()