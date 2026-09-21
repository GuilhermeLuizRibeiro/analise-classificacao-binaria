# Domínio: Saúde – Doença de Parkinson (base2_parkinson)

Como reproduzir (dentro desta pasta):

    pip install -r ../../requirements.txt
    python pipeline_parkinson.py        # split por sujeito, GridSearch, métricas, curvas ROC/PR, SHAP, probabilidades
    python aplicar_parte1_parkinson.py  # aplica o programa da Parte 1 (plot_corte) com t1=10% e t2=60%

Saídas em `resultados/`.
Base: `../data/base2_parkinson/base-dados-parkinson.csv` (vírgula como separador, ponto decimal restaurado; a versão original da UCI tinha os pontos decimais removidos).
