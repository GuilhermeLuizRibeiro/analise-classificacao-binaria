# Classificação binária com faixas de decisão - Avaliação A2

Trabalho da disciplina de Data Science. O projeto tem duas partes:

- **Parte 1 — gráfico de pontos de corte.** Um programa reutilizável que recebe os rótulos reais e as probabilidades previstas para a classe positiva e desenha um histograma (azul = todas as instâncias; vermelho = instâncias de rótulo real positivo). Com dois cortes `t1 < t2`, ele divide a população em três faixas de decisão e calcula os erros de cada uma.
- **Parte 2 — modelagem e relatório.** Classificação binária supervisionada em três domínios (sentimentos em avaliações, Parkinson por voz e sites de phishing), com três algoritmos por domínio, validação cruzada, métricas, curvas ROC/PR, SHAP beeswarm e aplicação da Parte 1 ao melhor modelo de cada domínio.

O relatório (`.docx`) é entregue separadamente.

## Sumário

1. [Estrutura do repositório](#1-estrutura-do-repositório)
2. [Instalação](#2-instalação)
3. [Bases de dados](#3-bases-de-dados)
4. [Como executar cada domínio](#4-como-executar-cada-domínio)
5. [Parte 1: como usar o programa](#5-parte-1-como-usar-o-programa)
6. [Resumo das decisões e resultados](#6-resumo-das-decisões-e-resultados)
7. [Reprodutibilidade](#7-reprodutibilidade)
8. [Problemas comuns](#8-problemas-comuns)

---

## 1. Estrutura do repositório

```
analise-classificacao-binaria/
├── requirements.txt
├── parte1_grafico_pontos_corte/
│   └── histograma_grafico_e_faixas.py        # programa da Parte 1
└── parte2_modelagem_relatorio/
    ├── data/
    │   ├── base1_reviews/                    # CSV + FONTE_LICENCA.MD
    │   ├── base2_parkinson/                  # CSV + FONTE_LICENCA.MD
    │   └── base3_phishing_websites/          # CSV + FONTE_LICENCA.MD
    ├── reviews/
    │   └── reviews.ipynb                     # domínio: sentimentos
    ├── parkinson/
    │   ├── pipeline_parkinson.py             # domínio: Parkinson (modelagem)
    │   ├── aplicar_parte1_parkinson.py       # domínio: Parkinson (Parte 1)
    │   ├── README.md
    │   └── resultados/                       # figuras, JSONs e probabilidades geradas
    └── phishing_websites/
        └── sites_de_phishing.ipynb           # domínio: phishing
```

Os domínios de sentimentos e phishing são **notebooks**. O domínio Parkinson é feito com **scripts Python**, e as saídas dele já estão versionadas em `parkinson/resultados/`.

---

## 2. Instalação

Requisitos: Python 3.10 ou superior (os notebooks foram executados com Python 3.14).

```bash
cd analise-classificacao-binaria

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

O `requirements.txt` já inclui Jupyter (`notebook`, `ipykernel`), `shap`, `scikit-learn`, `pandas`, `numpy` e `matplotlib`, todos com versão fixa.

> **Atenção à versão do scikit-learn.** O notebook de phishing usa `LogisticRegression(solver="saga")` com `l1_ratio` e sem `penalty`. Esse uso depende da versão do scikit-learn: em versões mais antigas o `l1_ratio` é ignorado quando `penalty` não é `elasticnet`. Os números salvos nos notebooks foram gerados com scikit-learn 1.8. Se o `pip` instalar outra versão, confira a seção [7](#7-reprodutibilidade).

---

## 3. Bases de dados

Todas as bases vêm do UCI Machine Learning Repository, sob licença **CC BY 4.0** (uso acadêmico permitido, com atribuição). A fonte completa de cada uma está em `FONTE_LICENCA.MD`, dentro da pasta da base.

| Domínio | CSV (em `parte2_modelagem_relatorio/data/`) | Alvo | Classe positiva (1) | Referência |
|---|---|---|---|---|
| Sentimentos em avaliações | `base1_reviews/sentiment_labelled_sentences.csv` | `class` | a frase expressa sentimento positivo | Kotzias (2015), DOI 10.24432/C57604 |
| Parkinson (voz) | `base2_parkinson/base-dados-parkinson.csv` | `status` | a pessoa tem Doença de Parkinson | Little (2007), DOI 10.24432/C59C74 |
| Sites de phishing | `base3_phishing_websites/base_phishing_sites.csv` | `Result` (recodificada: `target = Result == -1`) | o site é phishing | Mohammad e McCluskey (2012), DOI 10.24432/C51W2X |

| Base | Tamanho | Observações |
|---|---|---|
| Sentimentos | 3000 frases (Amazon, IMDb, Yelp) | colunas `sentence`, `source`, `class`; classes 50% / 50% |
| Parkinson | 195 gravações, 22 atributos numéricos, 32 sujeitos | a coluna `name` (`phon_R01_S##_#`) identifica o sujeito; 147 positivas e 48 negativas (75,4% / 24,6%) |
| Phishing | 11055 linhas, 30 atributos categóricos em {-1, 0, 1} | os notebooks removem 5206 duplicatas antes de dividir os dados, restando 5849 linhas |

---

## 4. Como executar cada domínio

Todos os caminhos são relativos. **Execute cada domínio a partir da pasta dele.**

### 4.1 Sentimentos — `reviews/reviews.ipynb`

```bash
cd parte2_modelagem_relatorio/reviews
jupyter lab            # abra reviews.ipynb e use Run All
```

Fluxo: proporção das classes → split estratificado 75/25 → busca de hiperparâmetros com `RandomizedSearchCV` (Naive Bayes, Regressão Logística, Random Forest; TF-IDF dentro do `Pipeline`) → seleção pelo F1 de validação → métricas, curvas e matrizes de confusão no teste → SHAP → Parte 1.

### 4.2 Phishing — `phishing_websites/sites_de_phishing.ipynb`

```bash
cd parte2_modelagem_relatorio/phishing_websites
jupyter lab            # abra sites_de_phishing.ipynb e use Run All
```

Fluxo: remoção de duplicatas → proporção das classes → split estratificado 75/25 → One-Hot Encoding dentro do `Pipeline` → busca de hiperparâmetros (Regressão Logística, Random Forest, Gradient Boosting; métrica-alvo **recall**) → métricas no teste → SHAP → Parte 1.

Nos dois notebooks, rode as células **na ordem**: a seleção do melhor modelo, o SHAP e a Parte 1 usam objetos criados nas células anteriores. A busca do Random Forest (35 combinações × 5 partições) é a etapa mais demorada.

### 4.3 Parkinson — scripts em `parkinson/`

```bash
cd parte2_modelagem_relatorio/parkinson
python pipeline_parkinson.py        # 1º: modelagem, curvas, SHAP e probabilidades
python aplicar_parte1_parkinson.py  # 2º: aplica a Parte 1 (t1 = 10%, t2 = 60%)
```

Os dois scripts precisam ser rodados **nesta ordem**: o segundo lê `resultados/probas_parkinson.npz`, gerado pelo primeiro.

Fluxo do `pipeline_parkinson.py`:

1. Extrai o sujeito da coluna `name`.
2. Separa o teste **por sujeito** com `StratifiedGroupKFold(5, shuffle=True, random_state=2)`, usando o primeiro fold como teste. Nenhum sujeito aparece ao mesmo tempo em desenvolvimento e teste, e o script confere isso com um `assert`.
3. Faz `GridSearchCV` (métrica-alvo F1, validação cruzada agrupada por sujeito) para Regressão Logística, Random Forest e SVM RBF, todos com `StandardScaler` no `Pipeline`.
4. Calcula as métricas no teste, gera as curvas ROC/PR e o SHAP beeswarm do melhor modelo.
5. Salva as probabilidades de validação (fora da amostra) e de teste para a Parte 1.
6. Calcula uma estimativa complementar por validação cruzada agrupada sobre todos os sujeitos.

Saídas em `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `results.json` | informações da base, resultados da CV, métricas de teste, melhor modelo, SHAP |
| `curvas_roc_pr.png` | curvas ROC e Precision-Recall dos três modelos no teste |
| `shap_beeswarm.png` | SHAP beeswarm do melhor modelo |
| `probas_parkinson.npz` | `y_val`, `p_val`, `y_test`, `p_test` (entrada da Parte 1) |
| `parte1_resultados.json` | tabelas e métricas das faixas, na validação e no teste |
| `figuras_parte1/` | histogramas de validação (escolha dos cortes) e de teste (resultado final) |

O CSV do Parkinson já está com o ponto decimal correto. O script ainda contém uma correção de escala para a versão da UCI, que veio sem ponto decimal, e ela só é aplicada se o arquivo estiver nesse formato.

---

## 5. Parte 1: como usar o programa

Arquivo: `parte1_grafico_pontos_corte/histograma_grafico_e_faixas.py`. Depende só de `numpy`, `pandas` e `matplotlib` e serve para qualquer modelo binário que produza a probabilidade da classe positiva (por exemplo, `predict_proba(X)[:, 1]` do scikit-learn).

```python
gerar_grafico_e_metricas(y_true, y_proba, bin_width=10, t1=None, t2=None)
```

| Parâmetro | Descrição |
|---|---|
| `y_true` | rótulos reais (0/1). **1 = classe positiva** do problema (ex.: "é phishing?") |
| `y_proba` | probabilidade estimada da **classe positiva para todas as instâncias**, e não só para as de rótulo positivo. Aceita fração (0–1) ou percentual (0–100) |
| `bin_width` | largura dos intervalos em pontos percentuais (padrão 10: 0–10%, 10–20%, …, 90–100%) |
| `t1`, `t2` | cortes em **percentual**, com `t1 < t2`. Se ambos forem informados, as métricas das faixas são calculadas |

Retorno: `(fig, ax, tabela_resumo, metricas)`. O item `metricas` é `None` quando não há cortes.

### Exemplo

```python
import sys
sys.path.append("parte1_grafico_pontos_corte")
from histograma_grafico_e_faixas import gerar_grafico_e_metricas

# 1) Validação: veja o histograma sem cortes e escolha t1 e t2
fig, ax, tabela, _ = gerar_grafico_e_metricas(y_val, p_val, bin_width=10)

# 2) Teste: aplique os cortes escolhidos, sem novos ajustes
fig, ax, tabela, metricas = gerar_grafico_e_metricas(
    y_test, p_test, bin_width=10, t1=10, t2=60
)
fig.savefig("histograma_teste.png", dpi=150)
print(metricas["faixas"]["analise_manual"])
print(metricas["proporcao_positivos_em_negativo_automatico"])
```

### O que o programa faz

- **Escala.** Se o maior valor de `y_proba` for ≤ 1, tudo é multiplicado por 100. Os cortes `t1` e `t2` sempre são informados em percentual.
- **Gráfico.** Eixo X: probabilidade estimada da classe positiva (0%–100%). Eixo Y: % das instâncias do conjunto avaliado. Barra **azul**: todas as instâncias do intervalo. Barra **vermelha**: instâncias de rótulo real positivo do intervalo. As duas usam o **mesmo denominador** (N total do conjunto). Linhas tracejadas marcam `t1` e `t2`.
- **Intervalos do histograma.** Cada instância pertence a exatamente um intervalo (`pd.cut` com intervalos fechados à direita e o primeiro incluindo 0%).
- **Faixas de decisão.** Negativo automático: `P < t1`. Análise manual: `t1 ≤ P < t2`. Positivo automático: `P ≥ t2`.

> Os intervalos do histograma são fechados à direita, enquanto as faixas de decisão são fechadas à esquerda em `t1` e `t2`. Uma instância com probabilidade exatamente igual a um corte (por exemplo, 10,0%) aparece na barra do intervalo anterior, mas conta na faixa seguinte. Isso só afeta valores exatamente iguais ao corte.

### Saídas

`tabela_resumo` tem uma linha por intervalo: `faixa`, `qtd_total_por_faixa`, `pct_populacao_por_faixa`, `qtd_positivos_por_faixa` e `pct_positivos_na_faixa`.

`metricas`, quando há `t1` e `t2`:

| Chave | Conteúdo | Denominador |
|---|---|---|
| `faixas[f]["qtd_total"]`, `["pct_populacao"]` | instâncias e % da população na faixa `f` | N total do conjunto |
| `faixas[f]["qtd_positivos"]`, `["qtd_negativos"]` | quantidade de positivos e negativos reais na faixa | — |
| `faixas[f]["proporcao_positivos_na_faixa"]`, `["proporcao_negativos_na_faixa"]` | composição da faixa | instâncias da faixa |
| `proporcao_positivos_em_negativo_automatico` | positivos reais classificados automaticamente como negativos (≈ taxa de falso negativo) | total de positivos reais |
| `proporcao_negativos_em_positivo_automatico` | negativos reais classificados automaticamente como positivos (≈ taxa de falso positivo) | total de negativos reais |
| `denominadores` | descrição textual de cada denominador | — |

As chaves de faixa `f` são `negativo_automatico`, `analise_manual` e `positivo_automatico`.

### Regras de uso exigidas pelo trabalho

1. Escolha `t1` e `t2` **com dados de validação**. Nos três domínios, as probabilidades de validação são obtidas fora da amostra (`cross_val_predict`) sobre o conjunto de desenvolvimento.
2. Aplique os cortes ao **conjunto de teste sem novos ajustes**. Só o resultado do teste entra na apresentação final.
3. Defina explicitamente a classe positiva de cada problema.

---

## 6. Resumo das decisões e resultados

| | Sentimentos | Parkinson | Phishing |
|---|---|---|---|
| Algoritmos comparados | Naive Bayes, Regressão Logística, Random Forest | Regressão Logística, Random Forest, SVM RBF | Regressão Logística, Random Forest, Gradient Boosting |
| Representação | TF-IDF | `StandardScaler` | One-Hot Encoding |
| Divisão treino/teste | 75/25 estratificada (`random_state=42`) | por sujeito, `StratifiedGroupKFold` (`random_state=2`) | 75/25 estratificada (`random_state=42`) |
| Validação | `StratifiedKFold`, 5 partições | `StratifiedGroupKFold`, 5 partições | `StratifiedKFold`, 5 partições |
| Busca de hiperparâmetros | `RandomizedSearchCV` | `GridSearchCV` | `RandomizedSearchCV` |
| Métrica-alvo | F1 da classe positiva | F1 da classe positiva | recall da classe positiva |
| Melhor modelo (pela validação) | Naive Bayes | SVM RBF | Gradient Boosting |
| Limiar das métricas de decisão | 0,50 | 0,50 | 0,50 |
| Cortes da Parte 1 (`t1` / `t2`) | 20% / 80% | 10% / 60% | 10% / 90% |
| SHAP | `LinearExplainer` (log-odds de sentimento positivo) | `KernelExplainer` (probabilidade de Parkinson) | `TreeExplainer` (saída do Gradient Boosting) |

Desempenho do melhor modelo de cada domínio no **conjunto de teste** (limiar 0,50), conforme as saídas salvas:

| Domínio | Acurácia | F1 | AUC-ROC |
|---|---|---|---|
| Sentimentos (Naive Bayes) | 0,8093 | 0,8091 | 0,8926 |
| Parkinson (SVM RBF) | 0,8140 | 0,8824 | 0,8683 |
| Phishing (Gradient Boosting) | 0,9522 | 0,9532 | 0,9923 |

Todos os modelos são avaliados com acurácia, precisão, recall, F1, AUC-ROC, AUC-PR, matriz de confusão e as curvas ROC e PR. A **AUC-PR por integração trapezoidal** e a **Average Precision (AP)** são reportadas em colunas separadas, porque são cálculos diferentes.

### Exemplo de aplicação da Parte 1 (Parkinson, teste, `t1 = 10%`, `t2 = 60%`)

O conjunto de teste tem 43 gravações (31 positivas e 12 negativas).

| Faixa | Instâncias | % da população (÷ 43) | Positivos | Negativos |
|---|---|---|---|---|
| Negativo automático (`P < 10%`) | 0 | 0,0% | 0 | 0 |
| Análise manual (`10% ≤ P < 60%`) | 7 | 16,3% | 1 | 6 |
| Positivo automático (`P ≥ 60%`) | 36 | 83,7% | 30 | 6 |

- Positivos classificados automaticamente como negativos: 0 de 31 (0%), com denominador igual ao total de positivos reais.
- Negativos classificados automaticamente como positivos: 6 de 12 (50%), com denominador igual ao total de negativos reais.

---

## 7. Reprodutibilidade

- **Sementes.** `random_state=42` em sentimentos e phishing (split, validação cruzada, buscas e modelos); `random_state=2` no split do Parkinson; `default_rng(42)` na amostragem do SHAP.
- **Versões.** As saídas salvas nos notebooks foram geradas com Python 3.14 e scikit-learn 1.8. O `requirements.txt` fixa as versões das bibliotecas, e você pode conferir a instalada com `python -c "import sklearn; print(sklearn.__version__)"`. Se ela for diferente de 1.8, os resultados podem mudar (veja o próximo item).
- **Parkinson.** O `StratifiedGroupKFold` com `shuffle=True` pode gerar divisões diferentes conforme a versão do scikit-learn, mesmo com a mesma semente. Confira o split esperado em `resultados/results.json`: 152 gravações / 25 sujeitos em desenvolvimento e 43 gravações / 7 sujeitos no teste (31 positivas e 12 negativas). Se o split for outro, os hiperparâmetros escolhidos e todas as métricas mudam.

---

## 8. Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `FileNotFoundError` ao ler um CSV | o notebook ou script não foi executado a partir da própria pasta | entre na pasta do domínio (seção 4) antes de executar |
| `FileNotFoundError: resultados/probas_parkinson.npz` | `aplicar_parte1_parkinson.py` foi executado antes do `pipeline_parkinson.py` | rode o pipeline primeiro |
| `ModuleNotFoundError: parte1_grafico_pontos_corte` (notebooks) | a estrutura de pastas foi alterada | mantenha as pastas originais; os notebooks adicionam `../..` ao `sys.path` |
| `ModuleNotFoundError: shap` (ou outra biblioteca) | dependências não instaladas | `pip install -r requirements.txt` |
| Números diferentes dos do relatório | versões diferentes de bibliotecas, principalmente scikit-learn | veja a seção 7 |
| Busca de hiperparâmetros demorada | Random Forest com 35 combinações × 5 partições; SHAP do Parkinson (`KernelExplainer`) | aguarde, ou reduza `n_iter` para testes rápidos |
| Gráfico não aparece | execução sem interface gráfica | use `%matplotlib inline` no notebook, ou salve com `fig.savefig(...)` |