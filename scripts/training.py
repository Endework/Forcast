#


import os
import warnings
import sys

import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Get url from DVC
import dvc.api

path2 = r"C:\Users\ende\Desktop\test\Data\\testfinal.csv"
path1 = r"C:\Users\ende\Desktop\test\Data\\trainfinal.csv"
repo ='https://github.com/Endework/Forcast'
version1 = 'v7'
version2= 'v6'
data_url2 = dvc.api.get_url(
    path = path2,
    repo = repo,
    rev = version2
)

data_url1 = dvc.api.get_url(
    path = path1,
    repo = repo,
    rev = version1
)
mlflow.set_experiment('new')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the AdSmartABdata csv file from remote repositery
    test = pd.read_csv(data_url2, sep=",")
    train = pd.read_csv(data_url1, sep=",")

    

    # Split the data into training and test sets. (0.75, 0.25) split
    #train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["Sales"], axis=1)
    test_x = test.drop(["Sales"], axis=1)
    train_y = train[["Sales"]]
    test_y = test[["Sales"]]

    

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        #Log data params
        mlflow.log_param('data_url', data_url1)
        mlflow.log_param('data_version', version1)
        mlflow.log_param('input_rows', train.shape[0])
        mlflow.log_param('input_cols', train.shape[1])


        mlflow.log_param('data_url', data_url2)
        mlflow.log_param('data_version', version2)
        mlflow.log_param('input_rows', test.shape[0])
        mlflow.log_param('input_cols', test.shape[1])

        #Log artifacts: columns used for modeling
        cols_x = pd.DataFrame(list(train_x.columns))
        cols_x.to_csv('features.csv', header = False, index = False)
        mlflow.log_artifact('features.csv')

        cols_y = pd.DataFrame(list(train_y.columns))
        cols_y.to_csv('targets.csv', header = False, index = False)
        mlflow.log_artifact('targets.csv')

        mlflow.sklearn.log_model(lr, "model")
