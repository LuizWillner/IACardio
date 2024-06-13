import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import os

CURRENT_DIRECTORY = os.path.dirname(__file__)
FILE_PATH = os.path.join(CURRENT_DIRECTORY, 'heart_2022_no_nans.csv')

# FUNÇÃO PARA RODAR E PRINTAR OS RESULTADOS DOS MODELOS
def evaluate_model(model, xtest, ytest):
   
    # Predict Test Data 
    ypred = model.predict(xtest)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(ytest, ypred)
    prec = metrics.precision_score(ytest, ypred)
    rec = metrics.recall_score(ytest, ypred)
    f1 = metrics.f1_score(ytest, ypred)
    kappa = metrics.cohen_kappa_score(ytest, ypred)

    # Calculate area under curve (AUC)
    ypred_proba = model.predict_proba(xtest)[::,1]
    fpr, tpr, _ = metrics.roc_curve(ytest, ypred_proba)
    auc = metrics.roc_auc_score(ytest, ypred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(ytest, ypred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


#Abrir o dataset
heart_data = pd.read_csv(FILE_PATH ,encoding='unicode_escape')

# print(heart_data.shape)

# VERIFICA SE EXISTEM VALORES NULOS NO DATASET
print("Null values: ", heart_data.isna().values.any())

# VERIFICA DISTRIBUIÇÃO DA CLASSE ALVO DO PROBLEMA (vemos aqui que o dataset está desbalanceado)
print(heart_data['HadHeartAttack'].value_counts())


###################### PRÉ-PROCESSAMENTO ##########################################

# REMOCAO DE ALGUMAS COLUNAS (Conferir se faz sentido excluir essas, se tem mais alguma a ser excluída e justificar o porquê da exclusão)

heart_data = heart_data.drop(columns=['State', 'RemovedTeeth', 'LastCheckupTime', 'ChestScan', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 
                                      'TetanusLast10Tdap', 'HeightInMeters', 'WeightInKilograms'])




# AJUSTE DE ALGUMAS VARIÁVEIS

# DIABETES: SOMENTE SIM OU NÃO
heart_data['HadDiabetes'].replace({'No, pre-diabetes or borderline diabetes' : 'Borderline', 'Yes, but only during pregnancy (female)' : 'During Pregnancy'} , inplace=True)

heart_data['HadDiabetes'].unique()

# SMOKER: SOMENTE SIM OU NÃO
heart_data['SmokerStatus'].replace({'Current smoker - now smokes some days' : 'Current smoker(Some days)',
                                    'Current smoker - now smokes every day' : 'Current smoker(Every day)'}, inplace=True)

heart_data['SmokerStatus'].unique()

# E-CIGARETTE: SOMENTE SIM OU NÃO
heart_data['ECigaretteUsage'].replace({'Not at all (right now)' : 'Not at all',
                                        'Never used e-cigarettes in my entire life' : 'Never',
                                        'Use them every day' : 'Everyday',
                                        'Use them some days' : 'Somedays'}, inplace=True)

heart_data['ECigaretteUsage'].unique()

# CRIANDO A COLUNA SMOKING_HABIT (Unifica SMOKER e E-CIGARETTE)

# Substituição dos valores nas colunas 'SmokerStatus' e 'ECigaretteUsage' por 'Smoker' se a pessoa fuma e 'Non-Smoker' se não fuma
heart_data['SmokerStatus'].replace({'Current smoker(Some days)': 'Smoker', 'Current smoker(Every day)': 'Smoker'}, inplace=True)
heart_data['ECigaretteUsage'].replace({'Everyday': 'Smoker', 'Somedays': 'Smoker'}, inplace=True)

# Nova coluna chamada 'SmokingHabit' que será 'Smoker' se a pessoa fuma em qualquer uma das duas colunas, caso contrário, será 'Non-Smoker'
heart_data['SmokingHabit'] = np.where((heart_data['SmokerStatus'] == 'Smoker') | (heart_data['ECigaretteUsage'] == 'Smoker'), 'Smoker', 'Non-Smoker')

# Remoção as colunas 'SmokerStatus' e 'ECigaretteUsage' já que agora elas estão representadas pela coluna 'SmokingHabit'
heart_data.drop(columns=['SmokerStatus', 'ECigaretteUsage'], inplace=True)


# RAÇA / ETNIA: REDUZIDA A 4 OPÇÕES
heart_data['RaceEthnicityCategory'].replace({'White only, Non-Hispanic' : 'White',
                                             'Black only, Non-Hispanic' : 'Black',
                                             'Other race only, Non-Hispanic' : 'Other Race',
                                             'Multiracial, Non-Hispanic' : 'Multi Racial'}, inplace=True)

heart_data['RaceEthnicityCategory'].unique()


# COVID: SOMENTE SIM OU NÃO
heart_data['CovidPos'].replace({'Tested positive using home test without a health professional' : 'Yes'}, inplace=True)

heart_data['CovidPos'].unique()

# FAIXA ETÁRIA: SIMPLIFICAÇÃO DAS VARIÁVEIS XX-XX ou 80+
heart_data['AgeCategory'].replace({'Age 80 or older' : '80+'}, inplace=True)

for value in heart_data['AgeCategory'].unique()[0:]:
    value = str(value)
    if value != 'nan' and value != '80+':
        cat_value = value.split(" ")
        heart_data['AgeCategory'].replace({value : cat_value[1]+"-"+cat_value[3]}, inplace=True)
        
heart_data['AgeCategory'].unique()

# PRINTA AS VARIÁVEIS DE CADA CLASSE APÓS AS SIMPLIFACAÇÕES
for col in heart_data.describe(include='object').columns:
    print('Column Name: ',col)
    print(heart_data[col].unique())
    print('-'*50)


# REMOVE ENTRADAS DUPLICADAS
heart_data.drop_duplicates(inplace=True)
print("duplicadas: ", (heart_data.shape))

from sklearn.preprocessing import OneHotEncoder

# Crie uma instância do codificador
encoder = OneHotEncoder()

# Ajuste e transforme as variáveis categóricas
encoded_data = encoder.fit_transform(heart_data[['Sex', 'RaceEthnicityCategory']])


# CONVERTE PARA ATRIBUTOS NUMERICOS
label=LabelEncoder()
for col in heart_data:
    heart_data[col]=label.fit_transform(heart_data[col])


# Calcular a matriz de correlação de Pearson
correlation_matrix = heart_data.corr(method='pearson')

# Plotar a matriz de correlação como um mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Matriz de Correlação de Pearson')
plt.show()


# separa classe alvo e as restantes
x = heart_data.drop('HadHeartAttack',axis=1)
y = heart_data['HadHeartAttack']

# SEPARA CONJUNTO DE TREINO E CONJUNTO  DE TESTE
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=.70,random_state=42) 


# Transforma o vetor de destino y para ter uma dimensão unidimensional
y_train = np.ravel(ytrain)


# PRINTA OS BOXPLOTS (PODEMOS USAR PARA IDENTIFICAR OUTLIERS)
""" for col in heart_data.describe().columns:
    plt.figure(figsize=(16, 2))  # Define o tamanho da figura como 16x2 polegadas
    sns.boxplot(data=heart_data, x=col)
    plt.show() """

# PRINTA OS GRAFICOS
""" gender_count = heart_data['Sex'].value_counts()
print(gender_count)

plt.title("Dsitribuição por Gênero", fontsize = 12, weight='bold')
plt.pie(gender_count,labels=gender_count.index,radius=1, autopct='%.2f%%')
plt.show()

# Mapeamento de renomeação das colunas
rename_mapping = {
    'Age 80 or older': '80+',
    'Age 75 to 79': '75 - 79',
    'Age 70 to 74': '70 - 74',
    'Age 65 to 69': '65 - 69',
    'Age 60 to 64': '60 - 64',
    'Age 55 to 59': '55 - 59',
    'Age 50 to 54': '50 - 54',
    'Age 45 to 49': '45 - 49',
    'Age 40 to 44': '40 - 44',
    'Age 35 to 39': '35 - 39',
    'Age 30 to 34': '30 - 34',
    'Age 25 to 29': '25 - 29',
    'Age 18 to 24': '18 - 24'
}

# Aplica a renomeação ao DataFrame
heart_data_renamed = heart_data.replace({'AgeCategory': rename_mapping})

# Define a ordem das categorias para o eixo x
order = ['18 - 24', '25 - 29', '30 - 34', '35 - 39', '40 - 44', '45 - 49', '50 - 54', '55 - 59', '60 - 64', '65 - 69', '70 - 74', '75 - 79', '80+']

# Define o estilo do seaborn e o tamanho do gráfico
sns.set(style='darkgrid')
plt.figure(figsize=(10, 5))

# Cria o gráfico de contagem com as colunas renomeadas e a ordem especificada
ax = sns.countplot(data=heart_data_renamed, x='AgeCategory', order=order)

# Função para personalizar o gráfico
def customize_plot(ax, title, xlabel, ylabel, title_fontsize, label_fontsize):
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)

# Chama a função para personalizar o gráfico
customize_plot(ax, "Distribuição por Idade", "Faixa Etária", "Indivíduos", 12, 10)

# Adiciona rótulos nas barras do gráfico
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=10)

plt.show()

# Define o estilo do Seaborn e o tamanho do gráfico
sns.set(style='darkgrid')
plt.figure(figsize=(12, 6))

# Cria o gráfico de contagem com a distribuição de ataques cardíacos por idade
ax = sns.countplot(data=heart_data_renamed, x='AgeCategory', hue='HadHeartAttack', palette='viridis')

# Função para personalizar o gráfico
def customize_plot(ax, title, xlabel, ylabel, title_fontsize, label_fontsize):
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)

# Chama a função para personalizar o gráfico
customize_plot(ax, "Incidência de Ataque Cardíaco por Faixa Etária", "Had Heart Attack", "Indivíduos", 12, 10)

# Adiciona rótulos nas barras do gráfico
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=10)

plt.show() """

""" # Separando os recursos (X) e o alvo (y)
X = heart_data.drop(columns=['HadHeartAttack'])
y = heart_data['HadHeartAttack']

from imblearn.under_sampling import ClusterCentroids

undersampler = ClusterCentroids(random_state=42)

X_resampled, y_resampled = undersampler.fit_resample(X, y)

print(pd.Series(y_resampled).value_counts()) """

# UNDERSAMPLING PARA BALANCEAMENTO DA CLASSE ALVO

from imblearn.under_sampling import RandomUnderSampler

# Separando os recursos (X) e o alvo (y)
X = heart_data.drop(columns=['HadHeartAttack'])
y = heart_data['HadHeartAttack']

# Instanciando o RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)

# Aplicando o undersampling aos dados
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Verificando a contagem de classes após o undersampling
print(pd.Series(y_resampled).value_counts())

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# K-Fold Cross-Validation
# DIVISÃO EM 10 FOLDS. PARA COMPARAÇÃO DOS MÉTODOS UTILIZADOS

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Modelos
models = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}


# Função para calcular métricas
def evaluate_model(model, X, y):
    metrics = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='weighted'),
        'roc_auc': make_scorer(roc_auc_score, average='weighted', needs_proba=True, multi_class='ovr')
    }
    
    results = {}
    for metric_name, metric in metrics.items():
        scores = cross_val_score(model, X, y, cv=kfold, scoring=metric)
        results[metric_name] = scores
    return results

# Avaliação dos Modelos com K-Fold
results = {}

for name, model in models.items():
    model_results = evaluate_model(model, X_scaled, y_resampled)
    results[name] = model_results
    print(f'{name}:')
    for metric_name, scores in model_results.items():
        print(f'  {metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})')

# Comparação Final dos Modelos
result_df = pd.DataFrame({metric: {model: results[model][metric].mean() for model in models} for metric in metrics})
print(result_df)
