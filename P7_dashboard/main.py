import streamlit as st
#st.set_page_config(layout="wide")
from app.preprocess import *
from app.model import *
from sklearn.metrics import f1_score, confusion_matrix,  recall_score, precision_score, fbeta_score
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import altair as alt
allow_output_mutation=True


st.sidebar.image('/app/files/logo.png')



baseUrl = 'https://p7-scoring-app.herokuapp.com/predict/'

@st.cache(persist=True)
def cache_data():
    return pd.read_csv('/app/files/data_df.csv', sep=';').drop(columns=['Unnamed: 0'])

@st.cache(persist=True)
def split(df):
    numerical_columns, _ = splitColsByType(df)
    X, y = create_X_Y(df, numerical_columns)
    return createTrainAndTestData(X, y)

@st.cache(persist=True)
def getModel(X_train, y_train):
    pickle_in = open("/app/files/classifier.pkl", "rb")
    classifier = pickle.load(pickle_in)
    return classifier

@st.cache(persist=True)
def getClassifiedData(skCurrId, threshold):
    url = baseUrl + '?skidCurr=' + str(skCurrId) +'&threshold=' + str(threshold)
    print('getClassifiedData')
    print(url)
    response = requests.get(url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    print('content: ',content)
    return content['proba']

@st.cache(persist=True)
def getSkIdList(df):
    return df.sample(100, random_state=1)['SK_ID_CURR']

def predictAll(df):
    pickle_in = open("/app/files/classifier.pkl", "rb")
    classifier = pickle.load(pickle_in)
    numerical_columns = np.delete(df.select_dtypes(['int64', 'float64']).columns, [0,1])
    X, _ = create_X_Y(df, numerical_columns)
    pred_proba = classifier.predict_proba(X)[:, 1]
    score_df = pd.DataFrame()
    score_df['SK_ID_CURR'] = df['SK_ID_CURR'].values
    score_df['SCORE'] = pred_proba
    return score_df

@st.cache(persist=True)
def predict(X_test, threshold):
    pickle_in = open("/app/files/classifier.pkl", "rb")
    classifier=pickle.load(pickle_in)
    print('classify datas....')
    pred_proba = classifier.predict_proba(X_test)[:, 1]
    pred_test = (pred_proba >= threshold).astype(bool)
    return pred_test

@st.cache(persist=True)
def getTop10MostImportantParameters(classifier, df):
    numerical_columns, _ = splitColsByType(df)
    X, _ = create_X_Y(df, numerical_columns)
    feature_imp = pd.DataFrame(sorted(zip(classifier.feature_importances_, X.columns)), columns=['Value', 'Feature'])
    return feature_imp.sort_values(by="Value", ascending=False)[:10]

@st.cache(persist=True)
def getColumnDescriptorData(cols, bureau_tableName, prevApp_tableName, app_tableName, ins_dict):
    colDescriptor_df = pd.read_csv("/app/files/HomeCredit_columns_description.csv", encoding = "ISO-8859-1").drop(columns=['Unnamed: 0'])
    visu_df = pd.DataFrame(columns=cols)
    row = {}
    for col in cols:
        if col.endswith('BUR'):
            table_df = colDescriptor_df[colDescriptor_df['Table'] == bureau_tableName]
            row[col] = table_df[table_df['Row'] == col[:-4]]['Description'].values[0]
        elif col.endswith('PREV_APP'):
            table_df = colDescriptor_df[colDescriptor_df['Table'] == prevApp_tableName]
            row[col] = table_df[table_df['Row'] == col[:-9]]['Description'].values[0]
        else:
            table_df = colDescriptor_df[colDescriptor_df['Table'] == app_tableName]
            if table_df[table_df['Row'] == col]['Description'].empty:
                row[col] = ins_dict[col]
            else:
                row[col] = table_df[table_df['Row'] == col]['Description'].values[0]
    return_df = visu_df.append(row, ignore_index=True).T.reset_index()
    return_df.columns = ['Column name', 'Description']
    return return_df

def fBetaScore(Precision, Recall, beta = 1):
    return ((1 + np.power(beta, 2)) * Precision * Recall) / ((np.power(beta, 2) * Precision) + Recall)

def plot_heatmap(pred_test, y_test):
    # Calculating and printing the f1 score
    cf_matrix = confusion_matrix(y_test, pred_test)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    fig, ax = plt.subplots()
    sns.set(rc={'figure.figsize': (15, 8)})
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Greens', cbar=False)
    sns.set(font_scale=1.4)
    precision = precision_score(y_test, pred_test, average='micro')
    recall = recall_score(y_test, pred_test)
    f1_test = fBetaScore(precision, recall, 2)
    return f1_test, recall, precision, fig

def getBarPlot(score, threshold):
    color_s = 'green'
    color_t = 'blue'
    if score > threshold:
        color_s = 'red'
    graphe_array = [['Score', score, color_s], ['Seuil',threshold, color_t ]]
    dataPlot = pd.DataFrame(graphe_array, columns=['Type', 'pourcentage', 'couleur'])
    fig, ax = plt.subplots()
    sns.set(font_scale=1.4)
    res = sns.barplot(dataPlot['Type'], dataPlot['pourcentage'], palette=dataPlot['couleur'])
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=18)
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width() / 2
        y = patches[i].get_height() + .05
        ax.annotate('{:.1f}%'.format(dataPlot['pourcentage'][i]*100), (x, y), ha='center')
    plt.ylim(0, 1)
    plt.title('Score du prêt vs seuil de décision', fontsize=20)
    return fig

def getMetricsBarPlot(f1, recall, precision):
    graphe_array = [['Moy entre rec. et prec.', f1], ['Détect. prêts risqués (recall)',recall], ['Détect. prêts sains (précision)', precision]]
    dataPlot = pd.DataFrame(graphe_array, columns=['Metrique', 'Score'])
    fig, ax = plt.subplots()
    sns.set(font_scale=1.4)
    res = sns.barplot(dataPlot['Metrique'], dataPlot['Score'])
    #res = sns.barplot(x=dataPlot['Metrique'], y=dataPlot['Score'], hue=dataPlot['Metrique'])
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=18)
    h, l = ax.get_legend_handles_labels()
    #ax.legend(h, ['f1', 'recall', 'precision'], title="Légende")
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width() / 2
        y = patches[i].get_height() + .05
        ax.annotate('{:.1f}%'.format(dataPlot['Score'][i]*100), (x, y), ha='center')
    plt.ylim(0, 1)
    plt.title('Score des différentes métriques (f2, recall, précision)', fontsize=20)
    return fig

def getThreshold():
    return st.sidebar.slider('Seuil de classification:', 0.0,1.0,0.5)

def getSkId(data):
    return st.sidebar.selectbox('Choix du sk-id:', getSkIdList(data), key=18)


def plotFeatureImportance(feature_importance):
    fig, ax = plt.subplots()
    sns.barplot(x="Value", y="Feature", data=feature_importance)
    plt.title('Top 10 des paramètres les plus influents pour le score')
    plt.tight_layout()
    return fig



def plotBarPlotAgainstQuantile(feature_imp, df, skCurrId, qnt):
    numerical_columns, _ = splitColsByType(df)
    X, _ = create_X_Y(df, numerical_columns)
    cols = feature_imp.sort_values(by="Value", ascending=False)[:10]
    print(cols)
    data_st = pd.DataFrame()
    data_st['SK_ID_CURR'] = df['SK_ID_CURR']
    row = {'SK_ID_CURR': skCurrId}
    for col in cols['Feature'].values:
        data_st[col] = X[col]
        row[col] = data_st[col].quantile(qnt)
    rowSerie = pd.Series(data=row)
    data_st_slct = data_st[data_st['SK_ID_CURR'] == skCurrId]
    data_st_slct = data_st_slct.append(rowSerie, ignore_index=True)
    data_st_slct.drop(columns=['SK_ID_CURR'], inplace=True)
    columns_raw = []
    values_raw = []
    type_raw = []
    for col in data_st_slct.columns:
        columns_raw.append(col)
        columns_raw.append(col)
        values_raw.append(data_st_slct[col][0])
        values_raw.append(data_st_slct[col][1])
        type_raw.append('Valeur pour le client')
        type_raw.append('quantile ' + str(qnt))
    df2 = pd.DataFrame({'column': columns_raw, 'values': values_raw, 'type': type_raw})
    fig, ax = plt.subplots()
    sns.set(style='white')
    sns.barplot(x='values', y='column', hue='type', data=df2)
    #data = pd.melt(df2.reset_index(), id_vars=["column"])
    return fig

df = cache_data()
ins_dict = {'DPD_INS': 'Days past due',
            'DBD_INS': 'Days before due',
            'PAYMENT_PERC_INS': 'Percentage of amount paid on previous installment',
            'PAYMENT_DIFF_INS': 'Difference between amount due and amount paid on previous installment'}

X_train, X_test, y_train, y_test = split(df)
classifier = getModel(X_train, y_train)
feature_importance = getTop10MostImportantParameters(classifier, df)
bureau_tableName = 'bureau.csv'
prevApp_tableName = 'previous_application.csv'
app_tableName = 'application_{train|test}.csv'
ins_tableName = 'installments_payments.csv'
colDescriptor_df = getColumnDescriptorData(feature_importance['Feature'].values, bureau_tableName=bureau_tableName,
                                           prevApp_tableName=prevApp_tableName, app_tableName=app_tableName,
                                           ins_dict=ins_dict)

st.sidebar.header('Choix du seuil: ')
threshold = getThreshold()
st.sidebar.header('Analyse par prêt: ')
skCurrId = getSkId(df)

st.write('## Scoring pour le seuil de ' +str(threshold * 100) + '%')
score = getClassifiedData(skCurrId, 0.5)
pred_test = predict(X_test, threshold)
f1_test, recall, precision, fig = plot_heatmap(pred_test, y_test)

if st.sidebar.checkbox("Représentation graphique des données du prêt", False):
    st.subheader("Visualisation graphique des données standardisées du prêt pour les 10 paramètres les plus influents")
    qnt = st.slider('Choisissez le quantile de comparaison:', 0.0,1.0,0.75)
    st.write(plotBarPlotAgainstQuantile(feature_importance, df, skCurrId, qnt))
    if st.checkbox("Description des 10 paramètres les plus influents"):
        AgGrid(colDescriptor_df, fit_columns_on_grid_load=True, height=320)
if st.sidebar.checkbox("Score du prêt et décision", False):
    predict_s = (score >= threshold)
    prediction = 'Accepté'
    if predict_s:
        prediction = 'Refusé'
    subheader = "Prévision pour le prêt " +str(skCurrId) + ": " + prediction
    st.subheader(subheader)
    st.write(getBarPlot(score, threshold))



st.sidebar.header('Analyse du modèle: ')
#if st.sidebar.checkbox("Display data", False):
#    st.subheader("Loan dataset")
#    st.write(df)

if st.sidebar.checkbox("Importance des données", False):
    st.subheader("Importances des données pour le modèle")
    st.write(plotFeatureImportance(feature_importance))
if st.sidebar.checkbox("Métriques du modèle", False):
    st.subheader("Métriques du modèle")
    st.write(getMetricsBarPlot(f1_test, recall, precision))

if st.sidebar.checkbox("Heatmap du modèle", False):
    st.subheader("Carte thermique (Heatmap)")
    st.write(fig)
#X_train, X_test, y_train, y_test = split(df)
#model = getModel(X_train, y_train)