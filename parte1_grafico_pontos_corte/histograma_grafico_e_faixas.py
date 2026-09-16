from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _para_percentual(y_proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(y_proba, dtype=float)
    if proba.size == 0:
        return proba
    if np.nanmax(proba) <= 1.0:
        return proba * 100.0
    return proba


def _gerar_bins(bin_width: int) -> np.ndarray:
    if not (0 < bin_width <= 100):
        raise ValueError("bin_width deve ser um inteiro entre 1 e 100.")

    edges = np.arange(0, 100 + bin_width, bin_width, dtype=float)
    edges = edges[edges <= 100]
    if edges[-1] < 100:
        edges = np.append(edges, 100.0)

    edges[-1] = edges[-1] + 1e-9
    return edges


def _rotulo_faixa(low: float, high: float) -> str:
    high_exibido = min(high, 100.0)
    return f"{low:.0f}%-{high_exibido:.0f}%"


def plot_corte(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    bin_width: int = 10,
    t1: Optional[float] = None,
    t2: Optional[float] = None,
) -> Tuple[Figure, Axes, pd.DataFrame, Optional[Dict[str, Any]]]:
    
    y_true_arr = np.asarray(y_true).astype(int)
    proba_bruta = np.asarray(y_proba)

    if y_true_arr.size != proba_bruta.size:
        raise ValueError("y_true e y_proba devem ter o mesmo tamanho.")
    if y_true_arr.size == 0:
        raise ValueError("y_true/y_proba nao podem ser vazios.")
    if not set(np.unique(y_true_arr)).issubset({0, 1}):
        raise ValueError("y_true deve conter apenas os valores 0 e 1.")

    proba_pct = _para_percentual(proba_bruta)
    n_total = proba_pct.size

    edges = _gerar_bins(bin_width)
    labels = [_rotulo_faixa(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    faixa = pd.cut(
        proba_pct,
        bins=edges,
        right=True,
        include_lowest=True,
        labels=labels,
    )

    df_base = pd.DataFrame({"faixa": faixa, "y_true": y_true_arr})

    contagem_total = (
        df_base.groupby("faixa", observed=False)["y_true"]
        .size()
        .reindex(labels, fill_value=0)
    )
    contagem_positivos = (
        df_base[df_base["y_true"] == 1]
        .groupby("faixa", observed=False)["y_true"]
        .size()
        .reindex(labels, fill_value=0)
    )

    pct_total = contagem_total / n_total * 100.0
    pct_positivos_sobre_total = contagem_positivos / n_total * 100.0
    contagem_total_np = contagem_total.to_numpy(dtype=float)
    contagem_positivos_np = contagem_positivos.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_positivos_na_faixa = np.where(
            contagem_total_np > 0,
            contagem_positivos_np / np.where(contagem_total_np == 0, 1, contagem_total_np) * 100.0,
            0.0,
        )

    tabela = pd.DataFrame(
        {
            "faixa": labels,
            "contagem_total": contagem_total.to_numpy(),
            "pct_total": pct_total.to_numpy(),
            "contagem_positivos": contagem_positivos.to_numpy(),
            "pct_positivos_na_faixa": pct_positivos_na_faixa,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    largura_barra = 0.4
    ax.bar(x - largura_barra / 2, pct_total.to_numpy(), width=largura_barra,
           color="tab:blue", label="Todas as instâncias (%)")
    ax.bar(x + largura_barra / 2, pct_positivos_sobre_total.to_numpy(), width=largura_barra,
           color="tab:red", label="Rótulo real positivo (%)")

    if t1 is not None and t2 is not None:
        for corte, nome in ((t1, "t1"), (t2, "t2")):
            pos = corte / bin_width - 0.5
            ax.axvline(pos, color="black", linestyle="--", linewidth=1)
            ax.text(pos, ax.get_ylim()[1] * 0.97, nome, rotation=90,
                    va="top", ha="right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
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

        def _resumo_faixa(mask: np.ndarray) -> Dict[str, Any]:
            n_faixa = int(mask.sum())
            pos_faixa = int((y_true_arr[mask] == 1).sum())
            neg_faixa = n_faixa - pos_faixa
            return {
                "contagem": n_faixa,
                "pct_populacao": (n_faixa / n_total * 100.0) if n_total else 0.0,
                "contagem_positivos": pos_faixa,
                "contagem_negativos": neg_faixa,
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
                "negativo_automatico": _resumo_faixa(mask_neg_auto),
                "analise_manual": _resumo_faixa(mask_manual),
                "positivo_automatico": _resumo_faixa(mask_pos_auto),
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
                    "contagem_faixa (numero de instancias daquela faixa)",
                "proporcao_positivos_em_negativo_automatico":
                    "total de positivos reais no conjunto avaliado (~ taxa de falso negativo)",
                "proporcao_negativos_em_positivo_automatico":
                    "total de negativos reais no conjunto avaliado (~ taxa de falso positivo)",
            },
        }

    return fig, ax, tabela, metricas