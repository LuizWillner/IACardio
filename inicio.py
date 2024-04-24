import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#Abrir o dataset
heart_data = pd.read_csv('/home/tatiana/IA/heart_2022_no_nans.csv' ,encoding='unicode_escape')

#heart_data.head()

print(heart_data.shape)

# print(heart_data.info())

# print(heart_data.isna().sum())

#all rows control for null values
print(heart_data.isna().values.any())

print(heart_data['HadHeartAttack'].value_counts())


# REMOCAO DE ALGUMAS COLUNAS
heart_data = heart_data.drop(columns=['State', 'RemovedTeeth', 'LastCheckupTime', 'ChestScan', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 
                                      'TetanusLast10Tdap', 'HighRiskLastYear'])




# AJUSTE DE ALGUMAS VARIÁVEIS

heart_data['HadDiabetes'].replace({'No, pre-diabetes or borderline diabetes' : 'Borderline', 'Yes, but only during pregnancy (female)' : 'During Pregnancy'} , inplace=True)

heart_data['HadDiabetes'].unique()

heart_data['SmokerStatus'].replace({'Current smoker - now smokes some days' : 'Current smoker(Some days)',
                                    'Current smoker - now smokes every day' : 'Current smoker(Every day)'}, inplace=True)

heart_data['SmokerStatus'].unique()

heart_data['ECigaretteUsage'].replace({'Not at all (right now)' : 'Not at all',
                                        'Never used e-cigarettes in my entire life' : 'Never',
                                        'Use them every day' : 'Everyday',
                                        'Use them some days' : 'Somedays'}, inplace=True)

heart_data['ECigaretteUsage'].unique()

heart_data['RaceEthnicityCategory'].replace({'White only, Non-Hispanic' : 'White',
                                             'Black only, Non-Hispanic' : 'Black',
                                             'Other race only, Non-Hispanic' : 'Other Race',
                                             'Multiracial, Non-Hispanic' : 'Multi Racial'}, inplace=True)

heart_data['RaceEthnicityCategory'].unique()




heart_data['CovidPos'].replace({'Tested positive using home test without a health professional' : 'Yes'}, inplace=True)

heart_data['CovidPos'].unique()

heart_data['AgeCategory'].replace({'Age 80 or older' : '80+'}, inplace=True)

for value in heart_data['AgeCategory'].unique()[1:]:
    value = str(value)
    if value != 'nan' and value != '80+':
        cat_value = value.split(" ")
        heart_data['AgeCategory'].replace({value : cat_value[1]+"-"+cat_value[3]}, inplace=True)
        
heart_data['AgeCategory'].unique()

# PRINTA AS VARIÁVEIS DE CADA CLASSE
""" for col in heart_data.describe(include='object').columns:
    print('Column Name: ',col)
    print(heart_data[col].unique())
    print('-'*50)
 """

#REMOVE ENTRADAS DUPLICADAS
heart_data.drop_duplicates(inplace=True)
# print(heart_data[heart_data.duplicated()])

# CONVERTE PARA ATRIBUTOS NUMERICOS
label=LabelEncoder()
for col in heart_data:
    heart_data[col]=label.fit_transform(heart_data[col])

# separa classe alvo e as restantes
x = heart_data.drop('HadHeartAttack',axis=1)
y = heart_data['HadHeartAttack']

# SEPARA CONJUNTO DE TREINO E CONJUNTO  DE TESTE
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=.70,random_state=42) 

# Transforma o vetor de destino y para ter uma dimensão unidimensional
y_train = np.ravel(ytrain)

# Create a KNN classifier object with 100 neighbors
knn = KNeighborsClassifier(n_neighbors=100)

# Train the classifier using the training data
knn.fit(xtrain, ytrain)

# Calculate the accuracy score on the test data
testing_score= knn.score(xtest, ytest)  # testing
print("Testing score:", testing_score)

# Calculate the accuracy score on the training data
training_score= knn.score(xtrain, ytrain)  # training
print("Training score:", training_score)

# print(heart_data)

# Display brief summary statistics
# print(heart_data.describe())

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