# Get url from DVC
# Import system libraries and our Scripts

import os
import sys
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join('../script')))
import dvc.api
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet


path1 =".. Data\trainfinal.csv"
Path2=".. Data\testfinal.csv"

repo="C:/Users/ende/Desktop/test"
version1='v2'
version2=v4
data_url1=dvc.api.get_url(path=path1,
repo=repo,rev=version1)
data_url2=dvc.api.get_ur2(path=path2,
repo=repo,rev=version2)
mlflow.set_experiment('expt')

def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2






if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    data1=pd.read_csv(data_url1,sep=",")
    data2=pd.read_csv(data_url2,sep=",")
    mlflow.log_param('data_version',version1)
    mlflow.log_param('data_url',data_url1)
    mlflow.log_param('input_rows',data1.shape[0])
    mlflow.log_param('input_cols',data1.shape[1])

    mlflow.log_param('data_version',version2)
    mlflow.log_param('data_url',data_url2)
    mlflow.log_param('input_rows',data2.shape[0])
    mlflow.log_param('input_cols',data2.shape[1])

    #train,test=train_test_split(data)
    train_x=data1.drop(['Sales'],axis=1)
    test_x=data2(["Sales"],axis=1)
    train_y=data1[['Sales']]
    test_y=data2[["Sales"]]


    cols_x=pd.DataFrame(list(train_x.columns))
    cols_x.to_csv('features.csv',header=False,index=False)
    mlflow.log_artifacts('feature.csv')


    cols_y=pd.DataFrame(list(train_y.columns))
    cols_x.to_csv('targer.csv',header=False,index=False)
    mlflow.log_artifacts('target.csv')


    alpha =float(sys.argv[1]) if len(sys.argv) >1 else 0.5
    l1_ratio=float(sys.argv[2]) if len(sys.argv) >2 else 0.5
    lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    lr.fit(train_x,train_y)






