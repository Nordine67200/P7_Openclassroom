import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from pathlib import Path

def load_data(path):
    dir = os.getcwd()
    print(dir)
    os.chdir(path)
    dir = os.getcwd()
    print(dir)
    print('loading data....')
    if os.path.exists('data_df.csv'):
        os.remove('data_df.csv')
    li_key_names = ['home_cred_desc', 'pos_cash_bal', 'app_test', 'app_train', 'bur',
                    'bur_bal', 'ccard_bal', 'install_pay', 'prev_app', 'samp_subm']
    dict_df = {}
    li_files = []
    path_data = ''
    for n_dir, _, n_files in os.walk(path_data):
        li_files = n_files
    li_files = sorted([s for s in li_files if '.csv' in s])
    for k, n_file in tqdm(zip(li_key_names, li_files)):
        dict_df[k] = pd.read_csv(path_data + '/' + n_file, encoding="ISO-8859-1")
    return dict_df


def cleanApplication(application_df):
    print('cleaning application data...')
    for col in tqdm(application_df.columns):
        prct = application_df[col].isna().sum() / application_df.shape[0]
        if prct >= 0.6:
            application_df.drop(columns=[col], inplace=True)
    return application_df


def createConsolidatedData(dict_df, test=False):
    print('creating the consolidated data....')
    appcol = 'app_train'
    if test:
        appcol = 'app_test'
    colonnes_app = ['SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE',
                    'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY',
                    'AMT_REQ_CREDIT_BUREAU_QRT']
    if test:
        colonnes_app = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE',
                        'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY',
                        'AMT_REQ_CREDIT_BUREAU_QRT']
    colonnes_bur = ['SK_ID_CURR', 'DAYS_CREDIT',
                    'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
                    'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM',
                    'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT']
    colonnes_prev_app = ['SK_ID_CURR', 'AMT_CREDIT', 'DAYS_LAST_DUE']
    # données app
    data_df = dict_df[appcol][colonnes_app]
    # données bureau
    bu_agg = {
        'DAYS_CREDIT': ['min'],  # nbre de jours avant la demande courante où le crédit a été demandé
        'CREDIT_DAY_OVERDUE': ['max'],  # nbre de jours d'impayés au moment de la demande courante
        'DAYS_CREDIT_ENDDATE': ['max'],  # nbre de jours restant
        'AMT_CREDIT_MAX_OVERDUE': ['max'],  # montant maximal d'impayés
        'AMT_CREDIT_SUM': ['sum'],  # montant actuel du crédit
        'AMT_CREDIT_SUM_DEBT': ['sum'],  # Dette actuelle du crédit
        'AMT_CREDIT_SUM_LIMIT': ['max']
    }
    bureau = dict_df['bur'][colonnes_bur]
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(bu_agg).reset_index()
    bureau_agg.columns = colonnes_bur
    data_df = pd.merge(data_df,
                       bureau_agg,
                       on='SK_ID_CURR',
                       how='left')
    # données prev_app
    prev_app_df = dict_df['prev_app'][colonnes_prev_app]
    dict_agg = {'AMT_CREDIT': ['sum'],
                'DAYS_LAST_DUE': ['max']
                }
    prev_app_aggr = prev_app_df.groupby('SK_ID_CURR').agg(dict_agg).reset_index()
    prev_app_aggr.columns = colonnes_prev_app
    data_df = pd.merge(data_df,
                       prev_app_aggr,
                       on='SK_ID_CURR',
                       how='left')
    colonnes_ins_pay = ['SK_ID_CURR',
                        'NUM_INSTALMENT_VERSION',
                        'AMT_PAYMENT',
                        'AMT_INSTALMENT',
                        'DAYS_ENTRY_PAYMENT',
                        'DAYS_INSTALMENT']
    ins = dict_df['install_pay'][colonnes_ins_pay]

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    ins = ins[['SK_ID_CURR', 'NUM_INSTALMENT_VERSION', 'PAYMENT_PERC', 'PAYMENT_DIFF', 'DPD', 'DBD']]
    ins_agg = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'PAYMENT_PERC': ['max'],
        'PAYMENT_DIFF': ['max'],
        'DPD': ['max'],
        'DBD': ['max']
    }
    ins_agg = ins.groupby('SK_ID_CURR').agg(ins_agg).reset_index()
    ins_agg.columns = ['SK_ID_CURR', 'NUM_INSTALMENT_VERSION', 'PAYMENT_PERC', 'PAYMENT_DIFF', 'DPD', 'DBD']
    data_df = pd.merge(data_df,
                       ins_agg,
                       on='SK_ID_CURR',
                       how='left')

    renamedColumns = ['SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                      'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE',
                      'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY',
                      'AMT_REQ_CREDIT_BUREAU_QRT', 'DAYS_CREDIT_BUR',
                      'CREDIT_DAY_OVERDUE_BUR', 'DAYS_CREDIT_ENDDATE_BUR',
                      'AMT_CREDIT_MAX_OVERDUE_BUR', 'AMT_CREDIT_SUM_BUR',
                      'AMT_CREDIT_SUM_DEBT_BUR', 'AMT_CREDIT_SUM_LIMIT_BUR',
                      'AMT_CREDIT_PREV_APP', 'DAYS_LAST_DUE_PREV_APP', 'NUM_INSTALMENT_VERSION_INS', 'PAYMENT_PERC_INS',
                      'PAYMENT_DIFF_INS', 'DPD_INS', 'DBD_INS']
    if test:
        renamedColumns = np.delete(renamedColumns, 1)
    data_df.columns = renamedColumns

    return data_df


def impute_bur_app_NaN(data):
    print('imputation of the bur and prev_app cols NaN....')
    for col in tqdm(data.columns):
        if col.endswith('BUR') or col.endswith('PREV_APP') or col.endswith('INS'):
            data[col] = data[col].fillna(0)
    data['AMT_REQ_CREDIT_BUREAU_QRT'] = data['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0)
    return data


def impute_NaN_By_Reg(df, list_col, target_col):
    print('imputation by regression....')
    lr = LinearRegression()
    testdf = df[df[target_col].isna()][list_col]
    traindf = df[~df[target_col].isna()][list_col]
    y = traindf[target_col]
    traindf.drop(target_col, axis=1, inplace=True)
    lr.fit(traindf, y)
    pred = lr.predict(df[list_col].drop(target_col, axis=1))
    df[target_col] = pred
    return df


def redefineType(data_df, test=False):
    print('redefining cols type')
    data_df['REGION_RATING_CLIENT_W_CITY'] = data_df['REGION_RATING_CLIENT_W_CITY'].astype('int32')
    if test == False:
        data_df['TARGET'] = data_df['TARGET'].astype('object')
    return data_df


def splitColsByType(data_df):
    print('splits cols by type.....')
    for col in data_df.columns:
        print(data_df[col].dtypes)
    num_cols = data_df.select_dtypes(['int64', 'float64']).columns.values
    numerical_columns = np.delete(num_cols, [0,1,2])
    categorical_columns = data_df.select_dtypes(['object']).columns.values
    return numerical_columns, categorical_columns


def delete_outliers(data_df):
    print('deleting outliers.....')
    data_df.drop(data_df[data_df['AMT_CREDIT_SUM_BUR'] > 0.8 * np.power(10, 8)].index, inplace=True)
    data_df.drop(data_df[data_df['AMT_CREDIT_MAX_OVERDUE_BUR'] > 5 * np.power(10, 6)].index, inplace=True)
    data_df.drop(data_df[data_df['AMT_CREDIT_SUM_DEBT_BUR'] > 4.5 * np.power(10, 7)].index, inplace=True)
    data_df.drop(data_df[data_df['AMT_CREDIT_SUM_DEBT_BUR'] == -6981558.210000001].index, inplace=True)
    data_df.drop(data_df[data_df['AMT_CREDIT_SUM_LIMIT_BUR'] > 2 * np.power(10, 6)].index, inplace=True)
    data_df.drop(data_df[data_df['AMT_CREDIT_PREV_APP'] > 2.6 * np.power(10, 7)].index, inplace=True)
    data_df.drop(data_df[data_df['PAYMENT_PERC_INS'] == np.inf].index, inplace=True)
    return data_df


def transform_cols(data_df, numerical_columns):
    print('power transformation of highly skewed cols')
    cols_to_transform = np.delete(numerical_columns, [3, 5, 13])
    dict_t_cols = {}
    for col in tqdm(cols_to_transform):
        print(col)
        t1 = PowerTransformer()
        p1 = t1.fit_transform(data_df[[col]] + np.abs(data_df[[col]].min()) + 1).flatten()
        dict_t_cols[col] = p1
    for col in tqdm(cols_to_transform):
        data_df[col] = dict_t_cols[col]
    return data_df


def harmonizeCategories(data_df):
    print('creating new categories fo categorical columns....')
    income_mapper = {
        'Working': 'Working',
        'Commercial associate': 'Commercial',
        'Businessman': 'Commercial',
        'Pensioner': 'Not working',
        'Unemployed': 'Not working',
        'Maternity leave': 'Not working',
        'Student': 'Not working',
        'State servant': 'State servant'

    }
    data_df['NAME_INCOME_TYPE'] = data_df['NAME_INCOME_TYPE'].map(income_mapper)
    education_mapper = {
        'Secondary / secondary special': 'Secondary / secondary special',
        'Higher education': 'Post secondary',
        'Incomplete higher': 'Post secondary',
        'Lower secondary': 'Lower secondary',
        'Academic degree': 'Post secondary'
    }
    data_df['NAME_EDUCATION_TYPE'] = data_df['NAME_EDUCATION_TYPE'].map(education_mapper)
    housing_mapper = {
        'House / apartment': 'House / apartment',
        'With parents': 'Other',
        'Municipal apartment': 'Rented appartment',
        'Rented apartment': 'Rented appartment',
        'Office apartment': 'Other',
        'Co-op apartment': 'Other'
    }
    data_df['NAME_HOUSING_TYPE'] = data_df['NAME_HOUSING_TYPE'].map(housing_mapper)
    return data_df

def preprocessing(dict_df, test=False):
    appcol = 'app_train'
    if test:
        appcol = 'app_test'
    dict_df[appcol] = cleanApplication(dict_df[appcol])
    data_df = createConsolidatedData(dict_df, test)
    data_df = impute_bur_app_NaN(data_df)
    list_col = ['AMT_CREDIT', 'AMT_ANNUITY']
    data_df = impute_NaN_By_Reg(data_df, list_col, 'AMT_ANNUITY')
    data_df = redefineType(data_df, test)
    numerical_columns, categorical_columns = splitColsByType(data_df)
    data_df = delete_outliers(data_df)
    #data_df = transform_cols(data_df, numerical_columns)
    data_df = harmonizeCategories(data_df)
    data_df.to_csv('data_df.csv', sep=';')
    return data_df, numerical_columns, categorical_columns