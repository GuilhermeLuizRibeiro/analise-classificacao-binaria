from typing import Tuple

import numpy as np
import os

from histograma_grafico_e_faixas import plot_corte


def _gerar_dados_classificador_sintetico(
    n: int, taxa_positivos: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    
    y = (rng.random(n) < taxa_positivos).astype(int)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    proba = np.empty(n, dtype=float)

    proba[y == 0] = rng.beta(2, 6, size=n_neg)

    proba[y == 1] = rng.beta(6, 2, size=n_pos)

    return y, proba * 100.0


if __name__ == "__main__":

    os.makedirs("imagens", exist_ok=True)

    rng = np.random.default_rng(42)
    taxa_positivos = 0.15

    y_val, proba_val = _gerar_dados_classificador_sintetico(3000, taxa_positivos, rng)
    y_teste, proba_teste = _gerar_dados_classificador_sintetico(3000, taxa_positivos, rng)

    fig_val, ax_val, tabela_val, _ = plot_corte(y_val, proba_val, bin_width=10)
    ax_val.set_title("Validacao -- usada so para escolher t1/t2")
    fig_val.savefig("imagens/validacao_escolha_corte.png", dpi=150)

    print("=== Tabela por faixa (VALIDACAO, usada so pra escolher os cortes) ===")
    print(tabela_val.to_string(index=False))

    t1, t2 = 25, 75

    fig_teste, ax_teste, tabela_teste, metricas = plot_corte(
        y_teste, proba_teste, bin_width=10, t1=t1, t2=t2
    )
    ax_teste.set_title(f"Teste -- resultado final, t1={t1} e t2={t2} travados na validacao")

    print("\n=== Tabela por faixa (TESTE, resultado final) ===")
    print(tabela_teste.to_string(index=False))

    print("\n=== Verificacao das 3 faixas de decisao (sanity check) ===")
    for nome, resumo in metricas["faixas"].items():
        print(f"{nome}: {resumo}")

    print("\nProporcao de positivos reais em negativo automatico "
          f"(falso negativo): {metricas['proporcao_positivos_em_negativo_automatico']:.1f}%")
    print("Proporcao de negativos reais em positivo automatico "
          f"(falso positivo): {metricas['proporcao_negativos_em_positivo_automatico']:.1f}%")

    fig_teste.savefig("imagens/teste_sanidade.png", dpi=150)
    print("\nGraficos salvos em validacao_escolha_corte.png e teste_sanidade.png")
