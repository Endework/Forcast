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
import os
import sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))

path2 ="dvcdata\\trainfinal.csv"
Path1="dvcdata\\testfinal.csv"

repo= ' /Users\ende\Desktop\\test'
#v6=testdata
#v7=traindata
version1='v6'
version2='v7'
data_url1=dvc.api.get_url(path=Path1,repo=repo,rev=version1)
data_url2=dvc.api.get_ur2(path=path2,repo=repo,rev=version2)
mlflow.set_experiment('expt')

def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2






if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    train=pd.read_csv(data_url2,sep=",")
    
    mlflow.log_param('data_version',version2)
    mlflow.log_param('data_url',data_url2)
    mlflow.log_param('input_rows',train.shape[0])
    mlflow.log_param('input_cols',train.shape[1])
    test=pd.read_csv(data_url1,sep=",")
    mlflow.log_param('data_version',version1)
    mlflow.log_param('data_url',data_url1)
    mlflow.log_param('input_rows',test.shape[0])
    mlflow.log_param('input_cols',test.shape[1])

    #train,test=train_test_split(data)
    train_x=train.drop(['Sales'],axis=1)
    test_x=test(["Sales"],axis=1)
    train_y=train[['Sales']]
    test_y=test[["Sales"]]


    cols_x=pd.DataFrame(list(train_x.columns))
    cols_x.to_csv('features.csv',header=False,index=False)
    mlflow.log_artifacts('feature.csv')


    cols_y=pd.DataFrame(list(train_y.columns))
    cols_x.to_csv('target.csv',header=False,index=False)
    mlflow.log_artifacts('target.csv')


    alpha =float(sys.argv[1]) if len(sys.argv) >1 else 0.5
    l1_ratio=float(sys.argv[2]) if len(sys.argv) >2 else 0.5
    lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    lr.fit(train_x,train_y)






