from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score,   recall_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle
import pandas as pd
import numpy as np


def OneHotEncoder(X,col, data_df):
    def categorise(x, category):
        if x == category:
            return 1
        else:
            return 0
    for category in list(set(data_df[col])):
        print('creating column',col + '_' + category)
        X[col + '_' + category] = X[col].apply(lambda x: categorise(x, category))
    return X.drop(columns=[col])


def create_X_Y(data_df, numerical_columns, df=pd.DataFrame()):
    print(df.shape)
    if df.shape[0] == 0:
        df = data_df
    X = data_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = data_df['TARGET'].values
    categorical_col_to_encode = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE']
    ss = StandardScaler()
    X[numerical_columns] = ss.fit_transform(X[numerical_columns])
    for col in categorical_col_to_encode:
        X = OneHotEncoder(X, col, df)
    y = y.astype('int')
    return X,y


def createTrainAndTestData(X,y):
    kf = RepeatedStratifiedKFold(n_splits=10, random_state=42)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def modelise(X_train, y_train):
    model_lgbm = LGBMClassifier(subsample=0.6000000000000001,
                                reg_lambda=5,
                                reg_alpha=7,
                                num_leaves=35,
                                min_child_weight=0.1,
                                min_child_samples=100,
                                colsample_bytree=0.4,
                                class_weight={0: 0.1, 1: 0.9})

    model_lgbm.fit(X_train, y_train)
    #pickle.dump(model_lgbm, open('../files/classifier.pkl', 'wb'))
    return model_lgbm

def predict(X_test, y_test, threshold):
    pickle_in = open("/app/files/classifier.pkl", "rb")
    classifier=pickle.load(pickle_in)
    print('classify datas...')
    pred_proba = classifier.predict_proba(X_test)[:, 1]
    pred_test = (pred_proba >= threshold).astype(bool)
    return pred_test, pred_proba




def getScores(y_pred, y_true):
    auc_score = roc_auc_score(y_true, y_pred)
    f1_test = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return auc_score, f1_test, recall, precision
