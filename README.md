# NLP_ML_Models

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://img.freepik.com/vetores-gratis/diagrama-de-circuito-de-tecnologia-azul-com-luzes-de-linha-brilhante_1017-17266.jpg?w=740&t=st=1686248413~exp=1686249013~hmac=59283fb41071ce03b79ae29079985dcd33a91aaece8b095f1ba9e50c938cdf35)

Um estudo dos modelos de processamento de linguagem natural utilizando machine learning e deep learning

## Introdução
O processamento de linguagem natural (PLN) tem se tornado cada vez mais relevante com o avanço da inteligência artificial. Com a capacidade de compreender e gerar linguagem humana, o PLN desempenha um papel fundamental em várias aplicações, desde assistentes virtuais até sistemas de tradução automática. Neste estudo, exploraremos diferentes modelos de PLN baseados em técnicas de machine learning e deep learning, buscando compreender suas vantagens e desafios.

## Metodologia
Para conduzir este estudo, utilizamos uma variedade de modelos de PLN populares, incluindo algoritmos clássicos de machine learning, como o Naive Bayes e o SVM (Support Vector Machine), e também redes neurais profundas, como as redes neurais convolucionais (CNN) e as redes neurais recorrentes (RNN). Coletamos conjuntos de dados relevantes, que abrangem diferentes tarefas de PLN, como classificação de sentimentos, identificação de entidades nomeadas e tradução automática.

## Treino modelo
Antes de treinar e avaliar os modelos, realizamos um pré-processamento dos dados. Essa etapa é crucial para melhorar a qualidade e a representação dos textos em formato adequado para os algoritmos de PLN. Durante o pré-processamento, aplicamos técnicas como tokenização, remoção de stopwords, lematização e normalização de palavras, visando reduzir a dimensionalidade e melhorar a eficiência dos modelos. Ao aplicar os modelos de PLN aos conjuntos de dados processados, avaliamos sua precisão, recall, F1-score e outras métricas relevantes, com o objetivo de comparar o desempenho de cada modelo em diferentes tarefas de PLN. Além disso, também exploramos as limitações e os desafios encontrados em cada abordagem.

## Modelos NLP

Os modelos de machine learning utilizados para processamento de linguagem natural (NLP, na sigla em inglês). Alguns dos modelos mais populares incluem:

A) Modelos baseados em redes neurais recorrentes (RNNs): Esses modelos são projetados para processar sequências de dados, como texto. Exemplos de RNNs comumente usadas em NLP incluem Long Short-Term Memory (LSTM) e Gated Recurrent Unit (GRU).

B) Modelos baseados em transformers: Os modelos transformers revolucionaram o campo de NLP. Esses modelos, como o BERT (Bidirectional Encoder Representations from Transformers) e o GPT (Generative Pre-trained Transformer), são pré-treinados em grandes quantidades de dados não supervisionados e, em seguida, afinados para tarefas específicas.

C) Redes neurais convolucionais (CNNs): Embora as CNNs sejam frequentemente associadas ao processamento de imagens, também podem ser usadas para processamento de texto. Esses modelos são especialmente úteis para tarefas de classificação de texto, como análise de sentimento e detecção de spam.

D) Modelos de aprendizado profundo para sequências: Além das RNNs, existem outros modelos de aprendizado profundo projetados especificamente para processar sequências, como as Redes Neurais de Transformadores (Transformer Neural Networks). Esses modelos combinam elementos das RNNs com a arquitetura de transformers para melhorar o desempenho em tarefas de NLP.

E) Modelos de aprendizado de máquina clássicos: Além dos modelos de aprendizado profundo, também existem algoritmos clássicos de aprendizado de máquina que podem ser usados para processamento de linguagem, como Naive Bayes, Máquinas de Vetores de Suporte (SVM) e Árvores de Decisão.

# Modelos de words embeddings

Os word embeddings que são amplamente utilizados em processamento de linguagem natural. Esses modelos são projetados para representar palavras como vetores numéricos em um espaço vetorial contínuo, onde palavras semanticamente similares são mapeadas para pontos próximos. Alguns dos modelos de word embeddings mais populares são:
Word2Vec: O Word2Vec é um modelo que aprende representações vetoriais densas para palavras, treinando uma rede neural em grandes quantidades de texto não rotulado. Ele oferece duas abordagens principais: Skip-gram e Continuous Bag-of-Words (CBOW). O Skip-gram prevê palavras de contexto dada uma palavra de entrada, enquanto o CBOW prevê uma palavra de entrada dado um contexto.

A) GloVe: O GloVe (Global Vectors for Word Representation) é um modelo de word embedding que combina a contagem global de co-ocorrência de palavras com técnicas de fatoração de matriz. Ele utiliza estatísticas de co-ocorrência para capturar relações semânticas e sintáticas entre palavras.

B) FastText: O FastText é uma extensão do Word2Vec que também leva em consideração subpalavras (n-gramas) durante o treinamento. Isso permite que o modelo lide melhor com palavras desconhecidas ou palavras fora do vocabulário.

C) ELMo: O ELMo (Embeddings from Language Models) é um modelo de word embedding contextual que captura a semântica da palavra em diferentes contextos. Ele usa um modelo de linguagem pré-treinado para gerar representações vetoriais contextualizadas para palavras.

D) BERT: O BERT (Bidirectional Encoder Representations from Transformers) é um modelo de linguagem pré-treinado baseado em transformers. Além de ser usado para tarefas de linguagem, o BERT também pode gerar word embeddings contextualizados que capturam as nuances semânticas das palavras em um contexto específico.

# Modelos análise de tópicos nlp

Os modelos utilizados para análise de tópicos em textos. Esses modelos são projetados para identificar e extrair automaticamente tópicos significativos em um conjunto de documentos. Alguns dos modelos de análise de tópicos mais populares são.

A) Latent Dirichlet Allocation (LDA): O LDA é um modelo generativo probabilístico que assume que os documentos são uma mistura de tópicos e que os tópicos são distribuições sobre palavras. Ele atribui probabilidades aos tópicos e palavras em cada documento, permitindo a identificação dos tópicos mais relevantes.

B) Non-Negative Matrix Factorization (NMF): O NMF é um modelo que fatora uma matriz não negativa em duas matrizes de menor dimensão. No contexto da análise de tópicos, a matriz pode representar a frequência de palavras em documentos. O NMF busca encontrar representações latentes que capturem tópicos distintos.

C) Correlated Topic Model (CTM): O CTM é uma extensão do LDA que permite capturar correlações entre tópicos. Ele assume que os tópicos estão correlacionados e que as palavras em cada documento são geradas a partir de uma mistura de tópicos correlacionados.

D) Hierarchical Dirichlet Process (HDP): O HDP é um modelo não paramétrico que estende o LDA para permitir um número indefinido de tópicos. Ele usa um processo de Dirichlet para inferir o número ideal de tópicos a partir dos dados.

E) Neural Topic Model (NTM): O NTM é um modelo baseado em redes neurais que combina elementos do LDA e de redes neurais. Ele usa uma arquitetura neural para inferir distribuições de tópicos em documentos.

## Conclusão
Neste estudo, analisamos diferentes modelos de processamento de linguagem natural utilizando técnicas de machine learning e deep learning. Observamos que tanto os algoritmos clássicos de machine learning, como Naive Bayes e SVM, quanto as redes neurais profundas, como CNNs e RNNs, apresentam resultados promissores em várias tarefas de PLN.

No entanto, também identificamos desafios enfrentados por esses modelos. O pré-processamento adequado dos dados mostrou-se fundamental para melhorar a qualidade das representações textuais e otimizar o desempenho dos modelos. Além disso, a disponibilidade de grandes volumes de dados rotulados é essencial para treinar modelos de deep learning de forma eficaz.

Portanto, concluímos que o PLN baseado em machine learning e deep learning tem um potencial significativo, mas ainda há espaço para melhorias. Pesquisas futuras devem explorar abordagens inovadoras, como transfer learning e modelos de linguagem pré-treinados, para aprimorar ainda mais o desempenho e a eficiência dos modelos de PLN.

## Autores

- [@RafaelGallo](https://github.com/RafaelGallo)


## Licença

[MIT](https://choosealicense.com/licenses/mit/)


## Resultados - Dos modelos machine learning 

- Melhorar o suporte de navegadores

- Adicionar mais integrações


## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

`API_KEY`

`ANOTHER_API_KEY`
## Instalação 

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib
  conda install keras
  conda install tensorflow-gpu==2.5.0

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
## Stack utilizada

**Machine learning:** Python, R

**Framework:** Scikit-learn

**Análise de dados:** Python, R

## Base de dados - Modelos de machine learning

| Dataset               | Link                                                 |
| ----------------- | ---------------------------------------------------------------- |
| Depression and Anxiety in Twitter| https://www.kaggle.com/datasets/stevenhans/depression-and-anxiety-in-twitter-id?select=datd_rand.csv|
| Email Spam Classification | https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv|
| Twitter Sentiment Analysis |https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis |

## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

Instalando a virtualenv

`pip install virtualenv`

Nova virtualenv

`virtualenv nome_virtualenv`

Ativando a virtualenv

`source nome_virtualenv/bin/activate` (Linux ou macOS)

`nome_virtualenv/Scripts/Activate` (Windows)

Retorno da env

`projeto_py source venv/bin/activate` 

Desativando a virtualenv

`(venv) deactivate` 

Instalando pacotes

`(venv) projeto_py pip install flask`

Instalando as bibliotecas

`pip freeze`

## Uso/Exemplos - Modelo machine learning

# Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib
  conda install nltk

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
  nltk==3.6.7
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
    
## Exemplo Modelo Word2Vec

```
## Python
# Importação das bibliotecas
from gensim.models import Word2Vec

# Corpus de exemplo
sentences = [['I', 'love', 'machine', 'learning'],
             ['I', 'love', 'deep', 'learning'],
             ['I', 'enjoy', 'natural', 'language', 'processing']]

# Treinando o modelo Word2Vec
model = Word2Vec(sentences, min_count=1)

# Obtendo o vetor de palavras
word_vectors = model.wv

# Obtendo o vetor de uma palavra específica
vector = word_vectors['machine']

# Encontrando palavras similares
similar_words = word_vectors.most_similar('learning')

# Imprimindo resultados
print("Vetor da palavra 'machine':")
print(vector)
print()

print("Palavras similares a 'learning':")
for word, similarity in similar_words:
    print(word, similarity)

## Modelo BERT em R
# Instalar os pacotes necessários
install.packages("huggingface")

# Carregar as bibliotecas
library(huggingface)

# Carregar o modelo pré-treinado BERT
model <- Huggingface_Model("bert-base-uncased")

# Texto de exemplo
texto <- "Eu adoro usar o modelo BERT para tarefas de processamento de linguagem natural."

# Tokenizar o texto
tokens <- model$tokenize(texto)

# Codificar os tokens
input_ids <- model$encode(tokens)

# Realizar a predição do modelo
output <- model$predict(input_ids)

# Imprimir os resultados
print(output)
```

## Feedback

Se você tiver algum feedback, por favor nos deixe saber por meio de rafaelhenriquegallo@gmail.com.br


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Melhorias

Que melhorias você fez no seu código? Ex: refatorações, melhorias de performance, acessibilidade, etc


## Referência

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Licença

[MIT](https://choosealicense.com/licenses/mit/)


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com
