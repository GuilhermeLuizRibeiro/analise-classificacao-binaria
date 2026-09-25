from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def _converter_para_percentual(y_proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(y_proba, dtype=float)
    if proba.size == 0:
        return proba
    if np.nanmax(proba) <= 1.0:
        return proba * 100.0
    return proba

def _definir_intervalos(bin_width: int) -> np.ndarray:
    if not (0 < bin_width <= 100):
        raise ValueError("bin_width deve ser um inteiro entre 1 e 100.")

    limites_faixas = np.arange(0, 100 + bin_width, bin_width, dtype=float)
    limites_faixas = limites_faixas[limites_faixas <= 100]
    if limites_faixas[-1] < 100:
        limites_faixas = np.append(limites_faixas, 100.0)

    limites_faixas[-1] = limites_faixas[-1] + 1e-9
    return limites_faixas

def _formatar_rotulo_da_faixa(low: float, high: float) -> str:
    high_exibido = min(high, 100.0)
    return f"{low:.0f}%-{high_exibido:.0f}%"

def gerar_grafico_e_metricas(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    bin_width: int = 10,
    t1: Optional[float] = None,
    t2: Optional[float] = None,
) -> Tuple[Figure, Axes, pd.DataFrame, Optional[Dict[str, Any]]]:
    
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_bruta = np.asarray(y_proba)

    if y_true_arr.size != y_proba_bruta.size:
        raise ValueError("y_true e y_proba devem ter o mesmo tamanho.")
    if y_true_arr.size == 0:
        raise ValueError("y_true/y_proba nao podem ser vazios.")
    if not set(np.unique(y_true_arr)).issubset({0, 1}):
        raise ValueError("y_true deve conter apenas os valores 0 e 1.")

    proba_pct = _converter_para_percentual(y_proba_bruta)
    n_total = proba_pct.size

    limites_faixas = _definir_intervalos(bin_width)
    rotulos_faixas = [_formatar_rotulo_da_faixa(limites_faixas[i], limites_faixas[i + 1]) for i in range(len(limites_faixas) - 1)]

    faixa = pd.cut(
        proba_pct,
        bins=limites_faixas,
        right=True,
        include_lowest=True,
        labels=rotulos_faixas,
    )

    df_base = pd.DataFrame({"faixa": faixa, "y_true": y_true_arr})

    qtd_total_por_faixa = (
        df_base.groupby("faixa", observed=False)["y_true"]
        .size()
        .reindex(rotulos_faixas, fill_value=0)
    )
    qtd_positivos_por_faixa = (
        df_base[df_base["y_true"] == 1]
        .groupby("faixa", observed=False)["y_true"]
        .size()
        .reindex(rotulos_faixas, fill_value=0)
    )

    pct_populacao_por_faixa = qtd_total_por_faixa / n_total * 100.0
    pct_positivos_sobre_populacao = qtd_positivos_por_faixa / n_total * 100.0
    qtd_total_por_faixa_np = qtd_total_por_faixa.to_numpy(dtype=float)
    qtd_positivos_por_faixa_np = qtd_positivos_por_faixa.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_positivos_na_faixa = np.where(
            qtd_total_por_faixa_np > 0,
            qtd_positivos_por_faixa_np / np.where(qtd_total_por_faixa_np == 0, 1, qtd_total_por_faixa_np) * 100.0,
            0.0,
        )

    tabela_resumo = pd.DataFrame(
        {
            "faixa": rotulos_faixas,
            "qtd_total_por_faixa": qtd_total_por_faixa.to_numpy(),
            "pct_populacao_por_faixa": pct_populacao_por_faixa.to_numpy(),
            "qtd_positivos_por_faixa": qtd_positivos_por_faixa.to_numpy(),
            "pct_positivos_na_faixa": pct_positivos_na_faixa,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rotulos_faixas))
    largura_barras = 0.4
    ax.bar(x - largura_barras / 2, pct_populacao_por_faixa.to_numpy(), width=largura_barras,
           color="tab:blue", label="Todas as instâncias (%)")
    ax.bar(x + largura_barras / 2, pct_positivos_sobre_populacao.to_numpy(), width=largura_barras,
           color="tab:red", label="Rótulo real positivo (%)")

    if t1 is not None and t2 is not None:
        for valor_corte, rotulo_corte in ((t1, "t1"), (t2, "t2")):
            pos = valor_corte / bin_width - 0.5
            ax.axvline(pos, color="black", linestyle="--", linewidth=1)
            ax.text(pos, ax.get_ylim()[1] * 0.97, rotulo_corte, rotation=90,
                    va="top", ha="right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(rotulos_faixas, rotation=45, ha="right")
    ax.set_xlabel("Probabilidade estimada da classe positiva")
    ax.set_ylabel("% das instâncias do conjunto avaliado")
    ax.set_title("Distribuição das probabilidades estimadas, por faixa")
    ax.legend()
    fig.tight_layout()

    metricas: Optional[Dict[str, Any]] = None
    if t1 is not None and t2 is not None:
        if t1 >= t2:
            raise ValueError("t1 deve ser menor que t2.")

        total_positivos = int((y_true_arr == 1).sum())
        total_negativos = int((y_true_arr == 0).sum())

        mask_neg_auto = proba_pct < t1
        mask_manual = (proba_pct >= t1) & (proba_pct < t2)
        mask_pos_auto = proba_pct >= t2

        def agregar_metricas_da_faixa(mask: np.ndarray) -> Dict[str, Any]:
            n_faixa = int(mask.sum())
            pos_faixa = int((y_true_arr[mask] == 1).sum())
            neg_faixa = n_faixa - pos_faixa
            return {
                "qtd_total": n_faixa,
                "pct_populacao": (n_faixa / n_total * 100.0) if n_total else 0.0,
                "qtd_positivos": pos_faixa,
                "qtd_negativos": neg_faixa,
                "proporcao_positivos_na_faixa": (pos_faixa / n_faixa * 100.0) if n_faixa else 0.0,
                "proporcao_negativos_na_faixa": (neg_faixa / n_faixa * 100.0) if n_faixa else 0.0,
            }

        positivos_em_neg_auto = int((y_true_arr[mask_neg_auto] == 1).sum())
        negativos_em_pos_auto = int((y_true_arr[mask_pos_auto] == 0).sum())

        metricas = {
            "t1": t1,
            "t2": t2,
            "n_total": n_total,
            "total_positivos": total_positivos,
            "total_negativos": total_negativos,
            "faixas": {
                "negativo_automatico": agregar_metricas_da_faixa(mask_neg_auto),
                "analise_manual": agregar_metricas_da_faixa(mask_manual),
                "positivo_automatico": agregar_metricas_da_faixa(mask_pos_auto),
            },
            "proporcao_positivos_em_negativo_automatico": (
                positivos_em_neg_auto / total_positivos * 100.0 if total_positivos else 0.0
            ),
            "proporcao_negativos_em_positivo_automatico": (
                negativos_em_pos_auto / total_negativos * 100.0 if total_negativos else 0.0
            ),
            "denominadores": {
                "pct_populacao": "N_total do conjunto avaliado (todas as instancias)",
                "proporcao_positivos_na_faixa / proporcao_negativos_na_faixa":
                    "qtd_total (numero de instancias daquela faixa)",
                "proporcao_positivos_em_negativo_automatico":
                    "total de positivos reais no conjunto avaliado (~ taxa de falso negativo)",
                "proporcao_negativos_em_positivo_automatico":
                    "total de negativos reais no conjunto avaliado (~ taxa de falso positivo)",
            },
        }

    return fig, ax, tabela_resumo, metricas