# Análise de Classificação Binária — Avaliação A2

Projeto da disciplina de Data Science com duas partes:

- **Parte 1 – Gráfico para análise de pontos de corte:** programa reutilizável que, a partir dos rótulos reais e das probabilidades estimadas para a classe positiva, gera um histograma (azul = todas as instâncias, vermelho = instâncias de rótulo positivo) e avalia três faixas de decisão definidas por dois cortes `t1 < t2`.
- **Parte 2 – Modelagem e relatório:** classificação binária supervisionada em três domínios, com três algoritmos por domínio, validação cruzada, métricas, curvas ROC/PR, SHAP (beeswarm) e aplicação da Parte 1 ao melhor modelo.

---

## Sumário

1. [Estrutura do repositório](#1-estrutura-do-repositório)
2. [Requisitos e instalação](#2-requisitos-e-instalação)
3. [Bases de dados, fontes e licenças](#3-bases-de-dados-fontes-e-licenças)
4. [Como executar a Parte 2 (notebooks)](#4-como-executar-a-parte-2-notebooks)
5. [Como usar a Parte 1 (programa dos histogramas)](#5-como-usar-a-parte-1-programa-dos-histogramas)
6. [Resumo das decisões por domínio](#6-resumo-das-decisões-por-domínio)
7. [Reprodutibilidade](#7-reprodutibilidade)
8. [Problemas comuns](#8-problemas-comuns)

---

## 1. Estrutura do repositório

```
analise-classificacao-binaria/
├── requirements.txt
├── parte1_grafico_pontos_corte/
│   └── histograma_grafico_e_faixas.py      # programa da Parte 1
└── parte2_modelagem_relatorio/
    ├── data/
    │   ├── base1_reviews/                  # sentiment_labelled_sentences.csv + FONTE_LICENCA.MD
    │   ├── base2_parkinson/                # base-dados-parkinson.csv + FONTE_LICENCA.MD
    │   └── base3_phishing_websites/        # base_phishing_sites.csv + FONTE_LICENCA.MD
    ├── reviews/reviews.ipynb                       # Domínio 1 – sentimentos
    ├── parkinson/parkinson.ipynb                   # Domínio 2 – Parkinson
    └── phishing_websites/sites_de_phishing.ipynb   # Domínio 3 – phishing
```

O relatório (`.docx`) é entregue separadamente.

---

## 2. Requisitos e instalação

- **Python 3.10 ou superior** (os notebooks foram executados com Python 3.14).
- Bibliotecas: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `shap` (listadas em `requirements.txt`) e **Jupyter** para abrir os notebooks (não listado no `requirements.txt`).
- Recomenda-se **scikit-learn ≥ 1.8**. O notebook de phishing usa `LogisticRegression` com `solver="saga"` e `l1_ratio` sem o parâmetro `penalty`, e versões mais antigas interpretam esses parâmetros de outra forma.

```bash
# 1. entrar na pasta do projeto
cd analise-classificacao-binaria

# 2. criar e ativar um ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows (PowerShell/cmd)

# 3. instalar as dependências
pip install -r requirements.txt
pip install jupyterlab             # ou: pip install notebook
```

---

## 3. Bases de dados, fontes e licenças

Todas as bases vêm do UCI Machine Learning Repository, sob licença **CC BY 4.0** (uso acadêmico permitido, com atribuição).

| Domínio | Arquivo CSV | Variável-alvo | Classe positiva (1) | Fonte |
|---|---|---|---|---|
| Sentimentos em avaliações | `base1_reviews/sentiment_labelled_sentences.csv` | `class` | a frase expressa sentimento positivo | Kotzias, D. (2015). *Sentiment Labelled Sentences*. DOI 10.24432/C57604 |
| Parkinson (voz) | `base2_parkinson/base-dados-parkinson.csv` | `status` | a pessoa tem Doença de Parkinson | Little, M. (2007). *Parkinsons*. DOI 10.24432/C59C74 |
| Sites de phishing | `base3_phishing_websites/base_phishing_sites.csv` | `Result` (recodificada: `target = Result == -1`) | o site é phishing | Mohammad, R.; McCluskey, L. (2012). *Phishing Websites*. DOI 10.24432/C51W2X |

Detalhes de cada base:

- **Sentimentos:** 3000 frases (Amazon, IMDb e Yelp), colunas `sentence`, `source`, `class`. Classes balanceadas (50%/50%).
- **Parkinson:** 195 gravações de voz de 32 identificadores de sujeito, 22 atributos numéricos, coluna `name` (padrão `phon_R01_S##_#`, de onde se extrai o sujeito) e `status`.
- **Phishing:** 11055 linhas e 30 atributos categóricos codificados em {-1, 0, 1}. Os notebooks removem as linhas duplicadas antes de dividir os dados (restam 5849).

---

## 4. Como executar a Parte 2 (notebooks)

Execute **cada notebook do início ao fim, na ordem das células** (`Run All`). As células dependem umas das outras: a seleção do melhor modelo, o SHAP e a aplicação da Parte 1 usam objetos criados antes.

> **Diretório de trabalho:** os notebooks usam caminhos relativos a partir da pasta onde estão salvos. Abra-os com o Jupyter iniciado na raiz do projeto (o Jupyter usa a pasta do notebook como diretório de trabalho). No VS Code, confirme que o diretório de trabalho do notebook é a pasta dele.

```bash
jupyter lab
# depois abra e execute:
#   parte2_modelagem_relatorio/reviews/reviews.ipynb
#   parte2_modelagem_relatorio/parkinson/parkinson.ipynb
#   parte2_modelagem_relatorio/phishing_websites/sites_de_phishing.ipynb
```

### 4.1 `reviews.ipynb` — sentimentos

Não exige ajustes. Lê `../data/base1_reviews/sentiment_labelled_sentences.csv` e importa a Parte 1 com `sys.path.append(os.path.abspath("../.."))`.

Fluxo: proporção das classes → split estratificado 75/25 → busca de hiperparâmetros (Naive Bayes, Regressão Logística, Random Forest; métrica-alvo F1; validação cruzada de 5 partições) → seleção pelo F1 de validação → métricas, curvas ROC/PR e matrizes de confusão no teste → SHAP → aplicação da Parte 1.

### 4.2 `sites_de_phishing.ipynb` — phishing

Não exige ajustes. Lê `../data/base3_phishing_websites/base_phishing_sites.csv`.

Fluxo: remoção de duplicatas → proporção das classes → split estratificado 75/25 → One-Hot Encoding dentro do `Pipeline` → busca de hiperparâmetros (Regressão Logística, Random Forest, Gradient Boosting; métrica-alvo **recall**) → métricas no teste → SHAP → aplicação da Parte 1.

### 4.3 `parkinson.ipynb` — Parkinson (**requer dois ajustes de caminho**)

Este notebook foi escrito para rodar em um ambiente tipo Google Colab, com o CSV e o módulo da Parte 1 na mesma pasta do notebook. No repositório eles estão em outras pastas, então **antes de executar** altere duas células:

**Célula 2 (carregamento):** troque o nome do arquivo.

```python
# antes
caminho_arquivo = 'base-dados-parkinson (2).csv'
# depois
caminho_arquivo = '../data/base2_parkinson/base-dados-parkinson.csv'
```

**Célula 1 (importações):** troque o import da Parte 1.

```python
# antes
from histograma_grafico_e_faixas import gerar_grafico_e_metricas

# depois
import sys, os
sys.path.append(os.path.abspath("../../parte1_grafico_pontos_corte"))
from histograma_grafico_e_faixas import gerar_grafico_e_metricas
```

*Alternativa (Google Colab):* envie para a sessão o CSV (`base-dados-parkinson.csv`, ajustando `caminho_arquivo` para esse nome) e o arquivo `histograma_grafico_e_faixas.py`, e execute sem alterar o import. A primeira célula já executa `!pip install -q shap`.

Fluxo: extração do sujeito a partir de `name` → split por sujeito com `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)` (o primeiro fold vira teste) → busca de hiperparâmetros com `GridSearchCV` (Regressão Logística, Random Forest, SVM RBF; métrica-alvo F1; validação cruzada agrupada por sujeito) → métricas no teste → SHAP (`KernelExplainer`) → aplicação da Parte 1.

**Conferência do split:** ao executar a célula 3, o esperado é

```
Instâncias de Desenvolvimento: 153 (25 sujeitos)
Instâncias de Teste Reservado  : 42 (7 sujeitos)
```

Se aparecerem outros números, veja a seção [7. Reprodutibilidade](#7-reprodutibilidade).

O `KernelExplainer` do SHAP é a parte mais lenta deste notebook.

---

## 5. Como usar a Parte 1 (programa dos histogramas)

Arquivo: `parte1_grafico_pontos_corte/histograma_grafico_e_faixas.py`. Depende apenas de `numpy`, `pandas` e `matplotlib`, e funciona com qualquer modelo binário que produza a probabilidade da classe positiva (por exemplo, `predict_proba(X)[:, 1]` do scikit-learn).

### Função principal

```python
gerar_grafico_e_metricas(y_true, y_proba, bin_width=10, t1=None, t2=None)
```

| Parâmetro | Descrição |
|---|---|
| `y_true` | rótulos reais (0/1); **1 = classe positiva** do problema (ex.: "é phishing?") |
| `y_proba` | probabilidade estimada da **classe positiva** para **todas** as instâncias, e não só para as de rótulo positivo. Aceita fração (0–1) ou percentual (0–100) |
| `bin_width` | largura dos intervalos em pontos percentuais (padrão 10, ou seja, 0–10%, 10–20%, …, 90–100%) |
| `t1`, `t2` | pontos de corte em **percentual**, com `t1 < t2`. Se ambos forem informados, as métricas das faixas de decisão são calculadas |

Retorno: `(fig, ax, tabela_resumo, metricas)`. `metricas` é `None` quando os cortes não são informados.

### Exemplo

```python
import numpy as np
from parte1_grafico_pontos_corte.histograma_grafico_e_faixas import gerar_grafico_e_metricas

# y_true: rótulos reais do CONJUNTO AVALIADO (0/1)
# y_proba: probabilidades da classe positiva desse mesmo conjunto
fig, ax, tabela, metricas = gerar_grafico_e_metricas(
    y_true, y_proba, bin_width=10, t1=20, t2=80
)
fig.savefig("histograma.png", dpi=150)
print(tabela)
print(metricas["faixas"]["analise_manual"])
print(metricas["proporcao_positivos_em_negativo_automatico"])
```

Para usar o módulo a partir de outra pasta, adicione a raiz do projeto ao `sys.path` (como fazem os notebooks) ou copie o arquivo `.py` para a pasta de trabalho.

### O que o programa faz

- **Escala:** se o maior valor de `y_proba` for ≤ 1, multiplica tudo por 100. Os cortes `t1` e `t2` devem ser informados sempre em percentual.
- **Intervalos:** cada instância pertence a exatamente um intervalo (`pd.cut` com intervalos fechados à direita e o primeiro incluindo 0%).
- **Gráfico:** eixo X = probabilidade estimada da classe positiva (0%–100%). Eixo Y = % das instâncias do conjunto avaliado. Barra **azul** = todas as instâncias do intervalo. Barra **vermelha** = instâncias de rótulo real positivo do intervalo. As duas usam o **mesmo denominador** (N total do conjunto). Linhas tracejadas marcam `t1` e `t2`.
- **Faixas de decisão:**
  - negativo automático: `P < t1`
  - análise manual: `t1 ≤ P < t2`
  - positivo automático: `P ≥ t2`

### Saídas

`tabela_resumo` (uma linha por intervalo): `faixa`, `qtd_total_por_faixa`, `pct_populacao_por_faixa`, `qtd_positivos_por_faixa`, `pct_positivos_na_faixa`.

`metricas` (quando há `t1` e `t2`):

| Chave | Conteúdo | Denominador |
|---|---|---|
| `faixas[...]["qtd_total"]`, `["pct_populacao"]` | instâncias e % da população em cada faixa | N total do conjunto |
| `faixas[...]["qtd_positivos"]`, `["qtd_negativos"]` | quantidade de positivos e negativos em cada faixa | — |
| `faixas[...]["proporcao_positivos_na_faixa"]`, `["proporcao_negativos_na_faixa"]` | composição da faixa | instâncias da faixa |
| `proporcao_positivos_em_negativo_automatico` | positivos classificados automaticamente como negativos (≈ taxa de falso negativo) | total de positivos reais |
| `proporcao_negativos_em_positivo_automatico` | negativos classificados automaticamente como positivos (≈ taxa de falso positivo) | total de negativos reais |
| `denominadores` | descrição textual de cada denominador | — |

As três faixas são `negativo_automatico`, `analise_manual` e `positivo_automatico`.

### Boas práticas de uso (exigidas pelo trabalho)

1. **Escolha `t1` e `t2` usando dados de validação.** Nos notebooks, as probabilidades de validação são obtidas por `cross_val_predict` (fora da amostra) sobre o conjunto de desenvolvimento. Chame primeiro a função **sem cortes** para ver o histograma e a tabela.
2. **Aplique os cortes ao conjunto de teste sem novos ajustes.** Só o resultado do teste entra na apresentação final.
3. Defina explicitamente a classe positiva de cada problema.

---

## 6. Resumo das decisões por domínio

| | Sentimentos | Parkinson | Phishing |
|---|---|---|---|
| Algoritmos comparados | Naive Bayes, Regressão Logística, Random Forest | Regressão Logística, Random Forest, SVM RBF | Regressão Logística, Random Forest, Gradient Boosting |
| Representação | TF-IDF | `StandardScaler` | One-Hot Encoding |
| Divisão treino/teste | 75/25 estratificada (`random_state=42`) | por sujeito, `StratifiedGroupKFold` (`random_state=2`) | 75/25 estratificada (`random_state=42`) |
| Validação | `StratifiedKFold` (5 partições) | `StratifiedGroupKFold` (5 partições) | `StratifiedKFold` (5 partições) |
| Busca de hiperparâmetros | `RandomizedSearchCV` | `GridSearchCV` | `RandomizedSearchCV` |
| Métrica-alvo | F1 da classe positiva | F1 da classe positiva | recall da classe positiva |
| Limiar das métricas de decisão | T = 0,50 | T = 0,50 | T = 0,50 |
| Cortes da Parte 1 (`t1` / `t2`) | 20% / 80% | 10% / 60% | 10% / 90% |
| SHAP | `LinearExplainer` (Naive Bayes) | `KernelExplainer` (SVM) | `TreeExplainer` (Gradient Boosting) |

Métricas calculadas para todos os modelos no conjunto de teste: acurácia, precisão, recall, F1, AUC-ROC, AUC-PR (integração trapezoidal), Average Precision (AP) e matriz de confusão, além das curvas ROC e Precision-Recall. A AUC-PR por trapézio e a AP são reportadas separadamente.

---

## 7. Reprodutibilidade

- Sementes fixas: `random_state=42` (reviews e phishing, split, validação cruzada, buscas e modelos), `random_state=2` no split do Parkinson e `default_rng(42)` na amostragem do SHAP.
- **Atenção no Parkinson:** o `StratifiedGroupKFold` com `shuffle=True` pode produzir divisões diferentes conforme a versão do scikit-learn, mesmo com a mesma semente. Numa execução de verificação com scikit-learn 1.8.0, o conjunto de teste saiu com 43 gravações (31 positivas e 12 negativas), e não com as 42 (18 positivas e 24 negativas) das saídas salvas no notebook. Nesse caso os hiperparâmetros escolhidos e todas as métricas mudam. Para reproduzir exatamente os números do relatório, use a mesma versão do scikit-learn com que os notebooks foram executados e confirme o "153 / 42" descrito na seção 4.3.
- Recomenda-se registrar a versão usada: `python -c "import sklearn; print(sklearn.__version__)"`.

---

## 8. Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `FileNotFoundError` ao ler um CSV | o notebook não está sendo executado a partir da própria pasta, ou (Parkinson) o nome/caminho do arquivo não foi ajustado | inicie o Jupyter com a pasta do notebook como diretório de trabalho e aplique os ajustes da seção 4.3 |
| `ModuleNotFoundError: histograma_grafico_e_faixas` (Parkinson) | o import não foi ajustado | aplique o ajuste de import da seção 4.3 |
| `ModuleNotFoundError: parte1_grafico_pontos_corte` (reviews/phishing) | a raiz do projeto não está no `sys.path` | mantenha a estrutura de pastas original; o notebook adiciona `../..` ao caminho |
| `ModuleNotFoundError: shap` | dependências não instaladas | `pip install -r requirements.txt` |
| Números diferentes dos do relatório | versões diferentes de bibliotecas (principalmente scikit-learn) | veja a seção 7 |
| A busca de hiperparâmetros demora | `RandomizedSearchCV` com Random Forest (35 combinações × 5 partições) e o SHAP do Parkinson são as etapas mais pesadas | aguarde ou reduza `n_iter` para testes rápidos |
| Gráfico não aparece | o notebook foi executado em modo sem interface | use `%matplotlib inline` ou salve com `fig.savefig(...)` |