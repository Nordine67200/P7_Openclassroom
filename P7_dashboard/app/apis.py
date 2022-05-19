from fastapi import FastAPI
from pydantic import BaseModel
from app.preprocess import load_data, preprocessing, splitColsByType
from app.model import modelise, create_X_Y, createTrainAndTestData, predict
import pandas as pd
import numpy as np
import os
from  app.utils import getDirectoryPath
from django.core.exceptions import SuspiciousOperation


app = FastAPI()



class InputParam(BaseModel):
    skidCurr: int
    threshold: float
    def __init__(self, x:int, y:float):
        self.skidCurr = x
        self.threshold = y


@app.get("/scoring/{id}")
def getScore(id:int):
    return{"score ": id}

#@app.get("/getXy")
#def getXy(df: str):
#    data_df = pd.read_json(df)
#    return create_X_Y(data_df, np.delete(data_df.select_dtypes(['int64', 'float64']).columns, [0,1]))


@app.get("/predict/")
def predictScore(skidCurr: int, threshold: float):
    print(os.getcwd())
    os.chdir('/files')
    dir = os.getcwd()
    pred_test = 0
    pred_proba = 0
    if (os.path.exists('data_df.csv')) & (os.path.exists('classifier.pkl')):
        print('Datas and model exist!')
        data_df = pd.read_csv('data_df.csv', sep=';')
        data_df.drop(columns=['Unnamed: 0'], inplace=True)
        X_test = data_df.drop(columns=['SK_ID_CURR', 'TARGET']).values
        y_test = data_df['TARGET'].values
        X_test, y_test = create_X_Y(data_df, np.delete(data_df.select_dtypes(['int64', 'float64']).columns, [0,1]))
        list_pred_test, list_pred_proba = predict(X_test, y_test, threshold)
        data_df['PRED_TARGET'] = list_pred_test
        data_df['PROBA_TARGET'] = list_pred_proba
        pred_test = data_df[data_df['SK_ID_CURR'] == skidCurr]['PRED_TARGET'].values[0]
        pred_proba = data_df[data_df['SK_ID_CURR'] == skidCurr]['PROBA_TARGET'].values[0]
        if pred_test:
            pred_test = 'Client risqué'
        else:
            pred_test = 'Client ok'
    else:
        raise SuspiciousOperation("Veuillez relancer le training!")
    return {'pred': pred_test, 'proba': pred_proba}

@app.post("/preprocess")
def preprocess(path: str = None):
    if path == None:
        path = getDirectoryPath('files')
    dict_df = load_data(path)
    preprocessing(dict_df)

@app.post("/modelise")
def modelise():
    os.chdir(getDirectoryPath('files'))
    data_df = pd.read_csv('data_df.csv', sep=';')
    numerical_columns, _ = splitColsByType(data_df)
    X, y = create_X_Y(data_df, numerical_columns)
    X_train, X_test, y_train, y_test = createTrainAndTestData(X, y)
    model = modelise(X_train, y_train)

@app.post("/processAndModelise")
def processAndModelise():
    dict_df = load_data(getDirectoryPath('files'))
    preprocessing(dict_df)


@app.get("/getConfigPathDirectoy")
def getPathDirectory():
    for root, dirs, files in os.walk("/"):
        for dir in dirs:
            print(dir)
            if dir == 'P7_scoring_config':
                return {'path': os.path.join(os.getcwd(), dir)}

