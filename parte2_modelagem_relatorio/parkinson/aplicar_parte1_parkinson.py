"""Aplica o programa da Parte 1 (plot_corte) ao melhor modelo do dominio Parkinson.
Entrada: resultados/probas_parkinson.npz gerado por pipeline_parkinson.py
  - validacao: probabilidades fora da amostra (CV agrupada) no conjunto de desenvolvimento -> escolha de t1/t2
  - teste: probabilidades do modelo final no conjunto reservado -> avaliacao com t1/t2 travados
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'parte1_grafico_pontos_corte'))
from histograma_grafico_e_faixas import plot_corte
R = 'resultados/'
os.makedirs(R + 'figuras_parte1', exist_ok=True)
d = np.load(R + 'probas_parkinson.npz')
# 1) Validacao: histograma/tabela por faixa para ESCOLHER t1 e t2
fig_v, ax_v, tab_v, _ = plot_corte(d['y_val'], d['p_val'], bin_width=10)
ax_v.set_title('Validação (probabilidades fora da amostra) - usada só para escolher t1/t2')
fig_v.savefig(R + 'figuras_parte1/parkinson_validacao_escolha_corte.png', dpi=150)
print(tab_v.round(1).to_string(index=False))
# 2) Cortes escolhidos na validacao (em %), travados antes de olhar o teste
T1, T2 = 10, 60
_, _, tab_vc, met_v = plot_corte(d['y_val'], d['p_val'], bin_width=10, t1=T1, t2=T2)
# 3) Teste: resultado final
fig_t, ax_t, tab_t, met_t = plot_corte(d['y_test'], d['p_test'], bin_width=10, t1=T1, t2=T2)
ax_t.set_title(f'Teste - resultado final, t1={T1}% e t2={T2}% travados na validação')
fig_t.savefig(R + 'figuras_parte1/parkinson_teste_resultado_final.png', dpi=150)
print(tab_t.round(1).to_string(index=False))
for nome, r in met_v['faixas'].items(): print('VAL', nome, r)
for nome, r in met_t['faixas'].items(): print('TESTE', nome, r)
print(met_t['proporcao_positivos_em_negativo_automatico'], met_t['proporcao_negativos_em_positivo_automatico'])
json.dump({'t1': T1, 't2': T2, 'val_tab': tab_v.to_dict('records'), 'test_tab': tab_t.to_dict('records'),
           'met_val': met_v, 'met_test': met_t}, open(R + 'parte1_resultados.json', 'w'), indent=1, default=float, ensure_ascii=False)
