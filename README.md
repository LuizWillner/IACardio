# Aplicação de modelo de ML para detecção de doenças cardíacas

Projeto da disciplina de Inteligência Artificial | prof. Taiane Ramos | 2024.1 | Universidade Federal Fluminense

## Descrição do projeto

### Objetivo

Este projeto visa desenvolver um modelo de Machine Learning capaz de prever, dada uma série de indicadores de saúde, se um indivíduo possui doenças cardíacas.

### Motivação

Doenças cardíacas são uma das principais causas de mortalidade em todo o mundo e sua detecção precoce junto a uma intervenção efetiva são cruciais para melhores resultados de tratamento do paciente.

- **Principal Causa de Morte**: [Segundo a OMS](https://www.who.int/health-topics/cardiovascular-diseases), as doenças cardíacas são a principal causa de morte globalmente, sendo responsáveis por cerca de 17,9 milhões de mortes anualmente. Sua detecção precoce junto a uma intervenção efetiva são cruciais para melhores resultados de tratamento do paciente.

- **Prevalência**: [Segundo o CDC](https://www.cdc.gov/heart-disease/about/index.html), Aproximadamente 1 em cada 5 mortes nos Estados Unidos é atribuída a doenças cardíacas.

- **Custo Econômico**: [Segundo o CDC](https://www.cdc.gov/heart-disease/about/index.html), somente nos EUA, as doenças cardíacas custam cerca de US$ 219 bilhões por ano em serviços de saúde, medicamentos e perda de produtividade.

### Base de dados utilizada

A base de dados foi obtida pela plataforma [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data?select=2022).

Esse conjunto vem originalmente do CDC e é uma parte importante do Sistema de Vigilância de Fatores de Risco Comportamentais (BRFSS), que realiza pesquisas telefônicas anuais para coletar dados sobre a saúde dos residentes dos EUA. Conforme descrito pelo CDC: "Criado em 1984 com 15 estados, o BRFSS agora coleta dados em todos os 50 estados, no Distrito de Columbia e em três territórios dos EUA. O BRFSS completa mais de 400.000 entrevistas com adultos a cada ano, tornando-o o maior sistema de pesquisa de saúde conduzido continuamente no mundo."

O conjunto de dados a ser utilizado **será a versão de 2022**. Mais especificamente, será utilizado o arquivo cujas ocorrências com valores "NaN" nas colunas foram removidas, a fim de facilitar a etapa de pré-processamento. Além disso, o autor da publicação no Kaggle notou diversos fatores oriundos das perguntas feitas na pesquisa que influenciam direta ou indiretamente as doenças cardíacas, tomando então a liberdade de selecionar as variáveis mais relevantes da base.

## Configurando ambiente de desenvolvimento

1. Baixar e instalar **Python 3.10**

   - No **Windows**, baixar e instalar pelo executável no [site](https://www.python.org/downloads/).
   - No **Ubuntu**, instalar pelo comando do terminal:
     ```shell
     >> sudo apt-get install python3.10
     ```

2. Criar ambiente virtual

   - No **Windows**:
     ```shell
     >> py -m venv .venv
     ```
   - No **Ubuntu**:
     ```shell
     >> python3 -m venv .venv
     ```

3. Ativar ambiente virtual. Sempre ativar quando ligar a máquina e iniciar o desenvolvimento.

   - No **Windows**:

     ```shell
     >> .\.venv/Scripts/activate
     ```

   - No **Ubuntu**:
     ```shell
     >> source .venv/bin/activate
     ```

4. Caso esteja no Windows, trocar politica de segurança do Windows, se necessário (executar comando abaixo no powershell como administrador):

   ```shell
   >> Set-ExecutionPolicy AllSigned
   ```

5. Instalar dependencies no venv.
   ```shell
   >> pip install -r requirements.txt
   ```
