# Classificação binária com faixas de decisão - Avaliação A2

Trabalho da disciplina de Data Science. O projeto tem duas partes:

- **Parte 1 - gráfico de pontos de corte.** Um programa reutilizável que recebe os rótulos reais e as probabilidades previstas para a classe positiva e desenha um histograma (azul = todas as instâncias; vermelho = instâncias de rótulo real positivo). Com dois cortes `t1 < t2`, ele divide a população em três faixas de decisão (negativo automático, análise manual, positivo automático) e calcula os erros de cada uma.
- **Parte 2 - modelagem e relatório.** Classificação binária supervisionada em três domínios (sentimento em avaliações, aprovação de anúncios de fazenda e sites de phishing), com três algoritmos por domínio, validação cruzada, métricas, curvas ROC/PR, SHAP beeswarm e aplicação da Parte 1 ao melhor modelo de cada domínio.

O relatório (`.docx`) acompanha a entrega e descreve a metodologia, os resultados e as conclusões. Este README documenta o **código**: como instalar, executar e reproduzir.

## Sumário

1. [Estrutura do repositório](#1-estrutura-do-repositório)
2. [Instalação](#2-instalação)
3. [Bases de dados, fontes e licenças](#3-bases-de-dados-fontes-e-licenças)
4. [Como executar cada domínio](#4-como-executar-cada-domínio)
5. [Metodologia comum aos três domínios](#5-metodologia-comum-aos-três-domínios)
6. [Parte 1: como usar o programa](#6-parte-1-como-usar-o-programa)
7. [Resumo dos resultados](#7-resumo-dos-resultados)
8. [Reprodutibilidade](#8-reprodutibilidade)
9. [Limitações conhecidas do código](#9-limitações-conhecidas-do-código)
10. [Problemas comuns](#10-problemas-comuns)
11. [Referências](#11-referências)

---

## 1. Estrutura do repositório

```
analise-classificacao-binaria/
├── README.md
├── requirements.txt
├── parte1_grafico_pontos_corte/
│   └── histograma_grafico_e_faixas.py        # programa da Parte 1
└── parte2_modelagem_relatorio/
    ├── data/
    │   ├── base1_reviews/                    # sentiment_labelled_sentences.csv + FONTE_LICENCA.MD
    │   ├── base2_anuncios_fazenda/           # farm-ads.csv + FONTE_LICENCA.MD
    │   └── base3_phishing_websites/          # base_phishing_sites.csv + FONTE_LICENCA.MD
    ├── reviews/
    │   └── reviews.ipynb                     # domínio: sentimento em avaliações
    ├── anuncios_fazenda/
    │   └── anuncios_fazenda.ipynb            # domínio: aprovação de anúncios (Farm Ads)
    └── phishing_websites/
        └── sites_de_phishing.ipynb           # domínio: sites de phishing
```

Cada domínio é um notebook autocontido: lê o CSV, treina e compara os três algoritmos, avalia no teste, gera o SHAP e aplica a Parte 1. Os notebooks importam o programa da Parte 1 diretamente da pasta `parte1_grafico_pontos_corte/`.

---

## 2. Instalação

Requisito: **Python 3.12 ou superior**. As versões fixas de `numpy`, `scipy` e `shap` exigem 3.12+, e os notebooks foram executados com Python 3.14.7.

```bash
cd analise-classificacao-binaria

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

O `requirements.txt` inclui as bibliotecas de modelagem (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `shap`) e o ambiente Jupyter (`notebook`, `jupyterlab`, `ipykernel`), todos com versão fixa.

> **Atenção à versão do scikit-learn.** O notebook de phishing usa `LogisticRegression(solver="saga")` com `l1_ratio` e **sem** `penalty`. Esse uso depende da versão: em versões antigas o `l1_ratio` é ignorado quando `penalty` não é `elasticnet`. Os números salvos nos notebooks foram gerados com **scikit-learn 1.8.0**. Confira a versão instalada com `python -c "import sklearn; print(sklearn.__version__)"`.

---

## 3. Bases de dados, fontes e licenças

As três bases vêm do UCI Machine Learning Repository, sob licença **CC BY 4.0** (uso acadêmico permitido, com atribuição). A fonte e a citação completas de cada uma estão no arquivo `FONTE_LICENCA.MD`, dentro da pasta da base.

| Domínio | CSV (em `parte2_modelagem_relatorio/data/`) | Coluna-alvo | Classe positiva (1) | Fonte (DOI) |
|---|---|---|---|---|
| Sentimento em avaliações | `base1_reviews/sentiment_labelled_sentences.csv` | `class` (`positive`/`negative`) | a frase expressa sentimento positivo | Kotzias (2015), [10.24432/C57604](https://doi.org/10.24432/C57604) |
| Anúncios de fazenda | `base2_anuncios_fazenda/farm-ads.csv` | `aprovado` (`1`/`-1`) | o anúncio foi aprovado pelo dono do conteúdo | Mesterharm e Pazzani (2011), [10.24432/C5ZC8D](https://doi.org/10.24432/C5ZC8D) |
| Sites de phishing | `base3_phishing_websites/base_phishing_sites.csv` | `Result` (`1` legítimo / `-1` phishing) | o site é phishing | Mohammad e McCluskey (2012), [10.24432/C51W2X](https://doi.org/10.24432/C51W2X) |

Em todos os notebooks a coluna `target` é criada com **1 = classe positiva** e **0 = classe negativa**:

| Domínio | Codificação do `target` |
|---|---|
| Sentimento | `target = 1` se `class == "positive"`, senão `0` |
| Farm Ads | `target = 1` se `aprovado == 1`, senão `0` (anúncio rejeitado) |
| Phishing | `target = 1` se `Result == -1` (phishing), senão `0` (legítimo) |

| Base | Tamanho | Distribuição das classes | Observações |
|---|---|---|---|
| Sentimento | 3.000 frases (Amazon, IMDb e Yelp, 1.000 de cada) | 1.500 positivas / 1.500 negativas (50% / 50%) | colunas `sentence`, `source`, `class` |
| Farm Ads | 4.143 anúncios de texto de 12 sites sobre animais de fazenda | 2.210 aprovados / 1.933 rejeitados (53,34% / 46,66%) | colunas `aprovado` e `texto` (o cabeçalho do CSV traz `" texto"` com espaço; o notebook remove os espaços dos nomes de colunas). O texto já vem pré-processado e marcado por prefixos de origem (`ad-`, `title-`, `header-`) |
| Phishing | 11.055 linhas, 30 atributos discretos em {-1, 0, 1} + `Result` | 6.157 legítimos / 4.898 phishing no original; **após remover 5.206 duplicatas: 5.849 linhas, 3.019 phishing / 2.830 legítimos (51,62% / 48,38%)** | 22 atributos binários e 8 ternários, tratados como categóricos |

---

## 4. Como executar cada domínio

Todos os caminhos são relativos. **Execute cada notebook a partir da pasta dele**, e rode as células **na ordem** (`Run All`): a seleção do melhor modelo, o SHAP e a Parte 1 usam objetos criados nas células anteriores.

```bash
# Sentimento
cd parte2_modelagem_relatorio/reviews
jupyter lab                 # abra reviews.ipynb e use Run All

# Anúncios de fazenda
cd parte2_modelagem_relatorio/anuncios_fazenda
jupyter lab                 # abra anuncios_fazenda.ipynb e use Run All

# Phishing
cd parte2_modelagem_relatorio/phishing_websites
jupyter lab                 # abra sites_de_phishing.ipynb e use Run All
```

Para executar sem interface (por exemplo, para conferir os resultados):

```bash
cd parte2_modelagem_relatorio/reviews
jupyter nbconvert --to notebook --execute reviews.ipynb --output /tmp/reviews_executado.ipynb
```

As etapas mais demoradas são as buscas de hiperparâmetros do Random Forest e do Gradient Boosting e o cálculo do SHAP (`TreeExplainer` com `interventional` no Farm Ads). Para um teste rápido, reduza o `n_iter` das buscas.

### Fluxo de cada notebook

| Etapa | Sentimento | Farm Ads | Phishing |
|---|---|---|---|
| 1. Leitura e `target` | `class` -> 0/1 | `aprovado == 1` -> 0/1 | remove duplicatas; `Result == -1` -> 0/1 |
| 2. Contagem das classes | quantidade e percentual por classe | idem | idem (após a deduplicação) |
| 3. Separação | `train_test_split` 75/25, estratificada, `random_state=42` (2.250 / 750) | idem (3.107 / 1.036) | idem (4.386 / 1.463) |
| 4. Validação | `StratifiedKFold(5, shuffle=True, random_state=42)` sobre o treino | idem | idem |
| 5. Representação (dentro do `Pipeline`) | TF-IDF, `token_pattern=r"\b[A-Za-zÀ-ÿ]+\b"` | idem | `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` -> 46 colunas |
| 6. Busca de hiperparâmetros | `RandomizedSearchCV`, métrica **F1** | `RandomizedSearchCV`, métrica **F1** | `GridSearchCV` (Reg. Logística) e `RandomizedSearchCV`, métrica **recall** |
| 7. Seleção | maior média na validação cruzada | idem | idem |
| 8. Avaliação no teste | métricas, curvas ROC/PR e matrizes de confusão (limiar 0,50) | idem | idem |
| 9. SHAP beeswarm | `LinearExplainer` | `TreeExplainer` | `TreeExplainer` |
| 10. Parte 1 | histograma da validação (probabilidades fora da amostra) e do teste com `t1`/`t2` fixos | idem | idem |

### Espaço de busca por algoritmo

| Domínio | Algoritmo | Busca | Combinações testadas (de quantas) |
|---|---|---|---|
| Sentimento | Naive Bayes Multinomial | `RandomizedSearchCV` | 400 (de 1.728) |
| Sentimento | Regressão Logística | `RandomizedSearchCV` | 400 (de 1.296) |
| Sentimento | Random Forest | `RandomizedSearchCV` | 60 (de 5.832) |
| Farm Ads | Regressão Logística | `RandomizedSearchCV` | 15 |
| Farm Ads | Random Forest | `RandomizedSearchCV` | 35 |
| Farm Ads | SGDClassifier | `RandomizedSearchCV` | 15 |
| Phishing | Regressão Logística | `GridSearchCV` | 28 (todas) |
| Phishing | Random Forest | `RandomizedSearchCV` | 40 (de 1.152) |
| Phishing | Gradient Boosting | `RandomizedSearchCV` | 50 (de 2.025) |

Os parâmetros do TF-IDF (`max_features`, `ngram_range`, `min_df`, `max_df`, `sublinear_tf`) fazem parte da busca nos domínios de texto. Os espaços completos estão nas células de cada notebook.

---

## 5. Metodologia comum aos três domínios

- **Teste reservado.** 25% da base, separado antes de qualquer ajuste, e consultado só na avaliação final. Os 75% restantes são o conjunto de desenvolvimento.
- **Validação cruzada no desenvolvimento.** `StratifiedKFold` de 5 partições, as mesmas para os três algoritmos. A média das 5 partições na métrica-alvo escolhe o melhor modelo.
- **Sem vazamento.** TF-IDF e One-Hot ficam dentro do `Pipeline`, então o vocabulário, os pesos IDF e as categorias são aprendidos só com as partes de treino de cada partição.
- **Limiar das métricas.** Acurácia, precisão, recall, F1 e matriz de confusão usam `LIMIAR = 0.5` (classe positiva se `P >= 0,5`). AUC-ROC, AUC-PR e AP não dependem de um limiar único.
- **Duas áreas sob a curva Precision-Recall, reportadas em colunas separadas:**
  - `AUC-PR (trapézio)`: integração trapezoidal dos pontos da curva (`sklearn.metrics.auc(recall, precisao)`).
  - `Average Precision (AP)`: `sklearn.metrics.average_precision_score`, soma ponderada sem interpolação linear.
- **Métrica-alvo.** F1 da classe positiva em sentimento e Farm Ads; recall da classe positiva em phishing, porque deixar passar um phishing (falso negativo) é o erro mais custoso. A justificativa completa está no relatório.

---

## 6. Parte 1: como usar o programa

Arquivo: `parte1_grafico_pontos_corte/histograma_grafico_e_faixas.py`. Depende só de `numpy`, `pandas` e `matplotlib` e serve para qualquer modelo binário que produza a probabilidade da classe positiva (por exemplo, `predict_proba(X)[:, 1]` do scikit-learn).

```python
gerar_grafico_e_metricas(y_true, y_proba, bin_width=10, t1=None, t2=None)
```

| Parâmetro | Descrição |
|---|---|
| `y_true` | rótulos reais, **0 ou 1**. **1 = classe positiva** do problema (ex.: "é phishing?") |
| `y_proba` | probabilidade estimada da **classe positiva para todas as instâncias**, e não só para as de rótulo positivo. Aceita fração (0-1) ou percentual (0-100) |
| `bin_width` | largura dos intervalos em pontos percentuais (padrão 10: 0%-10%, 10%-20%, ..., 90%-100%). Use um valor **inteiro que divida 100** (1, 2, 5, 10, 20, 25, 50) para obter intervalos e rótulos uniformes |
| `t1`, `t2` | cortes em **percentual** (ex.: `t1=30, t2=70`), com `t1 < t2`. Só calcula as métricas das faixas se **ambos** forem informados |

Retorno: `(fig, ax, tabela_resumo, metricas)`. O item `metricas` é `None` quando não há cortes. A função não chama `plt.show()` nem salva a figura; use `fig.savefig(...)`.

### Exemplo: probabilidades de um modelo do scikit-learn

```python
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

sys.path.append("parte1_grafico_pontos_corte")
from histograma_grafico_e_faixas import gerar_grafico_e_metricas

# 1) Validação: probabilidades FORA DA AMOSTRA no conjunto de desenvolvimento.
#    Veja o histograma sem cortes e escolha t1 e t2 com estes dados.
p_val = cross_val_predict(modelo, X_train, y_train, cv=validacao, method="predict_proba")[:, 1]
fig, ax, tabela_val, _ = gerar_grafico_e_metricas(y_train.values, p_val, bin_width=10)

# 2) Teste: aplique os cortes escolhidos, sem novos ajustes.
modelo.fit(X_train, y_train)
p_test = modelo.predict_proba(X_test)[:, 1]
fig, ax, tabela_teste, metricas = gerar_grafico_e_metricas(
    y_test.values, p_test, bin_width=10, t1=20, t2=80
)
fig.savefig("histograma_teste.png", dpi=150)
print(metricas["faixas"]["analise_manual"])
print(metricas["proporcao_positivos_em_negativo_automatico"])
```

Dentro dos notebooks, o import é feito adicionando a raiz do repositório ao `sys.path` (`sys.path.append(os.path.abspath("../.."))`) e importando `parte1_grafico_pontos_corte.histograma_grafico_e_faixas`.

### O que o programa faz

- **Escala.** Se o maior valor de `y_proba` for <= 1, tudo é multiplicado por 100; caso contrário, os valores são tratados como percentual. Os cortes `t1` e `t2` são **sempre** informados em percentual.
- **Gráfico.** Eixo X: probabilidade estimada da classe positiva (0%-100%), em intervalos de largura `bin_width`. Eixo Y: % das instâncias do conjunto avaliado. Barra **azul**: todas as instâncias do intervalo. Barra **vermelha**: instâncias de rótulo real positivo do intervalo. As duas usam o **mesmo denominador** (N total do conjunto) e há legenda para as duas. Linhas tracejadas marcam `t1` e `t2`.
- **Intervalos do histograma.** Cada instância pertence a exatamente um intervalo (`pd.cut` com intervalos fechados à direita; o primeiro inclui 0% e o último inclui 100%).
- **Faixas de decisão.**
  - Negativo automático: `P < t1`
  - Análise manual: `t1 <= P < t2`
  - Positivo automático: `P >= t2`

> Os intervalos do histograma são fechados à direita, enquanto as faixas de decisão são fechadas à esquerda em `t1` e `t2`. Uma instância com probabilidade exatamente igual a um corte (por exemplo, 10,0%) aparece na barra do intervalo anterior, mas conta na faixa de decisão seguinte. Isso só afeta valores exatamente iguais ao corte.

### Saídas

`tabela_resumo` tem uma linha por intervalo:

| Coluna | Conteúdo | Denominador |
|---|---|---|
| `faixa` | rótulo do intervalo (ex.: `10%-20%`) | - |
| `qtd_total_por_faixa` | instâncias no intervalo | - |
| `pct_populacao_por_faixa` | % da população no intervalo (barra azul) | N total do conjunto |
| `qtd_positivos_por_faixa` | positivos reais no intervalo | - |
| `pct_positivos_na_faixa` | % de positivos dentro do intervalo | instâncias do intervalo (`0.0` se o intervalo estiver vazio) |

`metricas`, quando há `t1` e `t2`:

| Chave | Conteúdo | Denominador |
|---|---|---|
| `t1`, `t2`, `n_total`, `total_positivos`, `total_negativos` | parâmetros e totais do conjunto | - |
| `faixas[f]["qtd_total"]`, `["pct_populacao"]` | instâncias e % da população na faixa `f` | N total do conjunto |
| `faixas[f]["qtd_positivos"]`, `["qtd_negativos"]` | positivos e negativos reais na faixa | - |
| `faixas[f]["proporcao_positivos_na_faixa"]`, `["proporcao_negativos_na_faixa"]` | composição da faixa | instâncias da faixa |
| `proporcao_positivos_em_negativo_automatico` | positivos reais classificados automaticamente como negativos (≈ taxa de falso negativo) | total de positivos reais |
| `proporcao_negativos_em_positivo_automatico` | negativos reais classificados automaticamente como positivos (≈ taxa de falso positivo) | total de negativos reais |
| `denominadores` | descrição textual de cada denominador | - |

As chaves de faixa `f` são `negativo_automatico`, `analise_manual` e `positivo_automatico`. Todos os valores `proporcao_*` e `pct_*` são percentuais (0-100).

### Validações feitas pelo programa

A função levanta `ValueError` quando: `y_true` e `y_proba` têm tamanhos diferentes; os vetores são vazios; `y_true` tem valores fora de {0, 1}; `bin_width` não está em (0, 100]; ou `t1 >= t2`.

### Regras de uso exigidas pelo trabalho

1. Escolha `t1` e `t2` **com dados de validação**. Nos três domínios, as probabilidades de validação são obtidas fora da amostra (`cross_val_predict`) sobre o conjunto de desenvolvimento.
2. Aplique os cortes ao **conjunto de teste sem novos ajustes**. Só o resultado do teste entra na apresentação final.
3. Defina explicitamente a classe positiva de cada problema (seção 3).

---

## 7. Resumo dos resultados

Valores salvos nos notebooks. Desempenho do melhor modelo de cada domínio no **conjunto de teste**, com limiar 0,50.

| | Sentimento | Farm Ads | Phishing |
|---|---|---|---|
| Algoritmos comparados | Naive Bayes, Regressão Logística, Random Forest | Regressão Logística, Random Forest, SGDClassifier | Regressão Logística, Random Forest, Gradient Boosting |
| Representação | TF-IDF | TF-IDF | One-Hot Encoding |
| Métrica-alvo | F1 da classe positiva | F1 da classe positiva | recall da classe positiva |
| Melhor modelo (pela validação) | Naive Bayes Multinomial | Random Forest | Gradient Boosting |
| Média da métrica-alvo na validação | 0,8185 | 0,9087 | 0,9541 |
| Teste: acurácia | 0,8213 | 0,8938 | 0,9515 |
| Teste: precisão | 0,8213 | 0,8686 | 0,9622 |
| Teste: recall | 0,8213 | 0,9439 | 0,9430 |
| Teste: F1 | 0,8213 | 0,9047 | 0,9525 |
| Teste: AUC-ROC | 0,8958 | 0,9696 | 0,9924 |
| Teste: AUC-PR (trapézio) / AP | 0,9051 / 0,9053 | 0,9730 / 0,9730 | 0,9934 / 0,9934 |
| SHAP (explicador) | `LinearExplainer` | `TreeExplainer` (`interventional`) | `TreeExplainer` (`tree_path_dependent`) |
| SHAP: saída explicada | log-odds de sentimento positivo | probabilidade de aprovação | log-odds de phishing |
| SHAP: dados explicados | os 750 exemplos do teste (fundo: 100 do treino) | 200 exemplos sorteados do teste (fundo: 100 do treino) | os 1.463 exemplos do teste |
| Cortes da Parte 1 (`t1` / `t2`) | 30% / 70% | 20% / 80% | 10% / 90% |

Hiperparâmetros do melhor modelo de cada domínio:

| Domínio | Hiperparâmetros |
|---|---|
| Sentimento (Naive Bayes) | `tfidf__max_features=3000`, `ngram_range=(1,1)`, `min_df=1`, `max_df=0.85`, `sublinear_tf=False`, `clf__alpha=1.0`, `clf__fit_prior=False` |
| Farm Ads (Random Forest) | `tfidf__max_features=500`, `ngram_range=(1,1)`, `min_df=2`, `max_df=0.9`, `sublinear_tf=True`, `clf__n_estimators=500`, `clf__max_depth=None`, `clf__min_samples_leaf=1` |
| Phishing (Gradient Boosting) | `clf__n_estimators=300`, `learning_rate=0.2`, `max_depth=4`, `subsample=1.0`, `min_samples_leaf=4`, `max_features="sqrt"` |

### Parte 1 no conjunto de teste (cortes escolhidos na validação)

| Domínio | Conjunto de teste | Negativo automático | Análise manual | Positivo automático |
|---|---|---|---|---|
| Sentimento (`30%` / `70%`) | 750 (375 pos. / 375 neg.) | 150 (20,0%): 9 pos. / 141 neg. | 434 (57,9%): 206 pos. / 228 neg. | 166 (22,1%): 160 pos. / 6 neg. |
| Farm Ads (`20%` / `80%`) | 1.036 (553 pos. / 483 neg.) | 298 (28,8%): 1 pos. / 297 neg. | 384 (37,1%): 204 pos. / 180 neg. | 354 (34,2%): 348 pos. / 6 neg. |
| Phishing (`10%` / `90%`) | 1.463 (755 pos. / 708 neg.) | 630 (43,1%): 13 pos. / 617 neg. | 178 (12,2%): 91 pos. / 87 neg. | 655 (44,8%): 651 pos. / 4 neg. |

Proporções de erro das decisões automáticas, com os denominadores:

| Domínio | Positivos classificados como negativos automáticos | Negativos classificados como positivos automáticos |
|---|---|---|
| Sentimento | 9 de 375 positivos reais = 2,4% | 6 de 375 negativos reais = 1,6% |
| Farm Ads | 1 de 553 positivos reais = 0,18% | 6 de 483 negativos reais = 1,24% |
| Phishing | 13 de 755 positivos reais = 1,72% | 4 de 708 negativos reais = 0,56% |

---

## 8. Reprodutibilidade

- **Sementes.** `random_state=42` em todos os domínios (separação treino/teste, `StratifiedKFold`, buscas e modelos). `np.random.default_rng(42)` na amostragem do fundo e dos exemplos do SHAP.
- **Versões.** As saídas salvas nos notebooks foram geradas com Python 3.14.7 e scikit-learn 1.8.0. O `requirements.txt` fixa as versões das bibliotecas. Com outra versão do scikit-learn, os resultados podem mudar (veja a atenção sobre o `l1_ratio` na seção 2).
- **Contagens esperadas.** Sentimento: 2.250 treino / 750 teste. Farm Ads: 3.107 / 1.036. Phishing: 5.206 duplicatas removidas (restam 5.849), 4.386 treino / 1.463 teste. Se os números forem outros, o ambiente ou os dados diferem do esperado.
- **Tempo.** O SHAP do Farm Ads usa 200 exemplos de teste e 100 de fundo com `TreeExplainer` `interventional`, e leva cerca de 1 minuto num computador comum.

---

## 9. Limitações conhecidas do código

Registradas aqui por transparência; algumas afetam a leitura dos resultados.

**Parte 1 (`histograma_grafico_e_faixas.py`)**

- Probabilidades `NaN`, negativas ou acima de 100% não entram em nenhum intervalo (`pd.cut` as descarta), embora continuem contadas em `n_total`. Nesse caso as barras não somam 100%. Passe apenas probabilidades válidas.
- A detecção de escala assume fração quando o máximo é <= 1. Uma lista de **percentuais** em que todos os valores sejam <= 1% (evento muito raro) seria interpretada como fração.
- `bin_width` fracionário (por exemplo, 2,5) gera rótulos arredondados (`0%-2%`, `2%-5%`), e uma largura que não divide 100 (por exemplo, 30) deixa o último intervalo mais estreito (`90%-100%`).
- Se apenas `t1` ou apenas `t2` for informado, os cortes são ignorados sem aviso.
- Cortes passados como fração (`0.3` e `0.7`) não são convertidos: são lidos como 0,3% e 0,7%.
- `y_true` precisa ser numérico em {0, 1}; texto como `"spam"`/`"ham"` deve ser convertido antes.
- Em intervalos vazios, `pct_positivos_na_faixa` vale `0.0`, e não "indefinido".

**Farm Ads**

- **SGDClassifier sem probabilidade.** O melhor SGD da busca usa `loss="hinge"`, que não tem `predict_proba`. O notebook usa então `decision_function`, e o limiar 0,50 é aplicado sobre essa margem, que não é uma probabilidade. AUC-ROC, AUC-PR e AP (baseadas em ranking) não são afetadas, mas acurácia, precisão, recall, F1 e a matriz de confusão do SGD não são comparáveis às dos outros dois modelos. O SGD não foi o modelo escolhido, e portanto não alimenta a Parte 1.
- **Textos duplicados.** A base tem 874 textos repetidos e o notebook não remove duplicatas (ao contrário do phishing). Na separação 75/25, 275 dos 1.036 textos do teste (26,5%) têm texto idêntico a um do treino. As métricas de teste podem estar otimistas.
- **Tokenização.** O regex `\b[A-Za-zÀ-ÿ]+\b` separa tokens como `ad-farm` em `ad` e `farm`, então os prefixos de origem (`ad-`, `title-`, `header-`) deixam de distinguir de onde a palavra veio.
- **Esforço de busca desigual.** Regressão Logística e SGD usam 15 combinações e o Random Forest usa 35. A diferença de F1 entre os três na validação é de 0,0016.

**Todos os domínios**

- As probabilidades "de validação" usadas para escolher `t1` e `t2` vêm de `cross_val_predict` com as mesmas partições usadas para escolher os hiperparâmetros. Isso é uma boa aproximação de dados nunca vistos, mas ligeiramente otimista.
- Os cortes `t1` e `t2` são valores fixos nas células finais dos notebooks. A comparação entre pares de cortes que fundamenta a escolha é feita no relatório, somando as faixas da tabela de validação.
- O Naive Bayes do domínio de sentimento produz probabilidades pouco calibradas (concentradas no meio da escala), o que reduz a cobertura das faixas automáticas.
- Sentimento tem 17 frases duplicadas na base; 3 frases do teste têm cópia idêntica no treino. Phishing foi deduplicado antes da separação, mas a base não permite saber se as duplicatas eram o mesmo site.

---

## 10. Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `FileNotFoundError` ao ler um CSV | o notebook não foi executado a partir da própria pasta | entre na pasta do domínio (seção 4) antes de abrir o Jupyter |
| `ModuleNotFoundError: parte1_grafico_pontos_corte` | a estrutura de pastas foi alterada | mantenha as pastas originais; os notebooks adicionam `../..` ao `sys.path` |
| `ModuleNotFoundError: shap` (ou outra biblioteca) | dependências não instaladas | `pip install -r requirements.txt` |
| `No matching distribution found` para `numpy==2.5.3` | Python anterior ao 3.12 | use Python 3.12 ou superior |
| Números diferentes dos do relatório | versões diferentes de bibliotecas, principalmente scikit-learn | veja as seções 2 e 8 |
| `NameError` em uma célula | células executadas fora de ordem | reinicie o kernel e use `Run All` |
| Busca de hiperparâmetros demorada | Random Forest e Gradient Boosting com muitas combinações × 5 partições | aguarde, ou reduza `n_iter` para testes rápidos |
| Gráfico não aparece | execução sem interface gráfica | use `%matplotlib inline` no notebook, ou salve com `fig.savefig(...)` |

---

## 11. Referências

- KOTZIAS, D. **Sentiment Labelled Sentences**. UCI Machine Learning Repository, 2015. DOI: 10.24432/C57604. KOTZIAS, D.; DENIL, M.; DE FREITAS, N.; SMYTH, P. From Group to Individual Labels Using Deep Features. KDD, 2015.
- MESTERHARM, C.; PAZZANI, M. **Farm Ads**. UCI Machine Learning Repository, 2011. DOI: 10.24432/C5ZC8D.
- MOHAMMAD, R.; MCCLUSKEY, L. **Phishing Websites**. UCI Machine Learning Repository, 2012. DOI: 10.24432/C51W2X. MOHAMMAD, R.; THABTAH, F.; MCCLUSKEY, L. An assessment of features related to phishing websites using an automated technique. International Conference for Internet Technology and Secured Transactions, 2012.
- LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 2017.