"""Domínio 1 (Saúde): Doença de Parkinson - pipeline com validação por sujeito."""
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve, auc, confusion_matrix)
import shap
SEED = 2; OUT = 'resultados/'
os.makedirs(OUT, exist_ok=True)
# ---------- 1. Dados ----------
df = pd.read_csv('../data/base2_parkinson/base-dados-parkinson.csv')
# O arquivo recebido perdeu o ponto decimal (ex.: 119,992 -> 11999200). Restaura a escala original da UCI:
_c = [c for c in df.columns if c not in ('name', 'status')]
_last6 = ['RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
if df['MDVP:Fo(Hz)'].max() > 1e4:  # arquivo sem ponto decimal
    for c in _c: df[c] = df[c] / (1e6 if c in _last6 else 1e5)
df['subject'] = df['name'].str.extract(r'(phon_R01_S\d+)')[0]
X = df.drop(columns=['name', 'status', 'subject']); y = df['status']; g = df['subject']
info = {'n': len(df), 'n_feat': X.shape[1], 'n_subj': int(g.nunique()),
        'subj_pd': int(df.groupby('subject').status.first().sum()),
        'rec_per_subj': [int(df.groupby('subject').size().min()), int(df.groupby('subject').size().max())],
        'counts': y.value_counts().to_dict()}
# ---------- 2. Split por sujeito (nenhum sujeito em treino e teste) ----------
outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
dev_idx, te_idx = next(outer.split(X, y, g))
Xd, yd, gd = X.iloc[dev_idx], y.iloc[dev_idx], g.iloc[dev_idx]
Xt, yt, gt = X.iloc[te_idx], y.iloc[te_idx], g.iloc[te_idx]
assert not set(gd) & set(gt)
info.update(dev_n=len(Xd), test_n=len(Xt), dev_subj=int(gd.nunique()), test_subj=int(gt.nunique()),
            dev_counts=yd.value_counts().to_dict(), test_counts=yt.value_counts().to_dict(),
            test_subj_pd=int(df.iloc[te_idx].groupby('subject').status.first().sum()))
# ---------- 3. Modelos + GridSearch (F1, CV agrupada) ----------
cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
cw = [None, 'balanced']
models = {
 'Regressão Logística': (LogisticRegression(max_iter=5000), {'m__C': [.01, .1, 1, 10], 'm__class_weight': cw}),
 'Random Forest': (RandomForestClassifier(random_state=SEED), {'m__n_estimators': [100, 200], 'm__max_depth': [3, 5, 10, None], 'm__class_weight': cw}),
 'SVM (RBF)': (SVC(kernel='rbf', probability=True, random_state=SEED), {'m__C': [.1, 1, 10], 'm__gamma': ['scale', 'auto', .1], 'm__class_weight': cw}),
}
fits, rows, cvrows = {}, [], []
for n, (m, grid) in models.items():
    pipe = Pipeline([('sc', StandardScaler()), ('m', m)])
    gs = GridSearchCV(pipe, grid, scoring='f1', cv=cv, n_jobs=-1, refit=True)
    gs.fit(Xd, yd, groups=gd)
    b = gs.best_index_
    cvrows.append({'Modelo': n, 'F1 (CV)': gs.best_score_, 'Desvio': gs.cv_results_['std_test_score'][b],
                   'Hiperparâmetros': {k[3:]: v for k, v in gs.best_params_.items()}})
    p = gs.predict_proba(Xt)[:, 1]; pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    pr, rc, _ = precision_recall_curve(yt, p)
    rows.append({'Modelo': n, 'Acurácia': accuracy_score(yt, pred), 'Precisão': precision_score(yt, pred), 'Recall': recall_score(yt, pred),
                 'F1': f1_score(yt, pred), 'AUC-ROC': roc_auc_score(yt, p), 'AUC-PR (trapz)': auc(rc, pr),
                 'AP': average_precision_score(yt, p), 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
    fits[n] = (gs, p)
cvdf = pd.DataFrame(cvrows); tdf = pd.DataFrame(rows)
best = cvdf.sort_values('F1 (CV)', ascending=False).iloc[0]['Modelo']
print(cvdf.to_string()); print(tdf.round(4).to_string()); print('MELHOR (validação):', best)
# ---------- 4. Curvas ROC e PR ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for n, (gs, p) in fits.items():
    f, t, _ = roc_curve(yt, p); ax[0].plot(f, t, label=f'{n} (AUC={roc_auc_score(yt, p):.3f})')
    pr, rc, _ = precision_recall_curve(yt, p); ax[1].plot(rc, pr, label=f'{n} (AUC-PR={auc(rc, pr):.3f}; AP={average_precision_score(yt, p):.3f})')
ax[0].plot([0, 1], [0, 1], 'k--', lw=.8, label='Aleatório'); ax[0].set(xlabel='FPR (1 - especificidade)', ylabel='TPR (recall)', title='Curva ROC (teste)')
ax[1].axhline(yt.mean(), color='k', ls='--', lw=.8, label=f'Prevalência ({yt.mean():.2f})'); ax[1].set(xlabel='Recall', ylabel='Precisão', title='Curva Precision-Recall (teste)', ylim=(0, 1.02))
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + 'curvas_roc_pr.png', dpi=170); plt.close()
# ---------- 5. Melhor modelo: SHAP beeswarm (conjunto de teste, saída = P(Parkinson)) ----------
gs, pt = fits[best]; est = gs.best_estimator_
Xt_s = pd.DataFrame(est['sc'].transform(Xt), columns=X.columns, index=Xt.index)
Xd_s = pd.DataFrame(est['sc'].transform(Xd), columns=X.columns)
mdl = est['m']
if isinstance(mdl, RandomForestClassifier):
    sv = shap.TreeExplainer(mdl).shap_values(Xt_s); sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv); out = 'probabilidade da classe 1'
elif isinstance(mdl, LogisticRegression):
    sv = shap.LinearExplainer(mdl, Xd_s).shap_values(Xt_s); out = 'log-odds da classe 1'
else:
    f = lambda d: mdl.predict_proba(d)[:, 1]
    np.random.seed(SEED); sv = shap.KernelExplainer(f, shap.kmeans(Xd_s, 20)).shap_values(Xt_s, nsamples=300); out = 'probabilidade da classe 1'
plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=12); plt.title(f'SHAP beeswarm - {best} (teste, {len(Xt)} instâncias)\nsaída explicada: {out}', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'shap_beeswarm.png', dpi=170, bbox_inches='tight'); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
shp = [{'feat': f, 'mean_abs': float(imp[f]), 'corr': float(np.corrcoef(Xt[f], sv[:, list(X.columns).index(f)])[0, 1])} for f in imp.index[:8]]
print(pd.DataFrame(shp).round(3)); print('saida:', out)
# ---------- 6. Probabilidades para a Parte 1 (validacao = OOF agrupado no desenvolvimento; teste = reservado) ----------
oof = cross_val_predict(est, Xd, yd, groups=gd, cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1), method='predict_proba')[:, 1]
np.savez(OUT + 'probas_parkinson.npz', y_val=yd.values, p_val=oof, y_test=yt.values, p_test=pt)
band = {}
# ---------- 7. Estimativa complementar: CV agrupada externa (todos os sujeitos), F1-otimizado com os hiperparâmetros escolhidos ----------
from sklearn.base import clone
oo = np.zeros(len(X)); 
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 7).split(X, y, g):
    e = clone(est).fit(X.iloc[tr], y.iloc[tr]); oo[te] = e.predict_proba(X.iloc[te])[:, 1]
pp = (oo >= .5).astype(int)
sup = {'roc': roc_auc_score(y, oo), 'ap': average_precision_score(y, oo), 'rec': recall_score(y, pp), 'prec': precision_score(y, pp), 'f1': f1_score(y, pp),
       'spec': ((pp == 0) & (y.values == 0)).sum() / (y.values == 0).sum()}
print(sup)
json.dump({'info': info, 'cv': cvrows, 'test': rows, 'best': best, 'shap': shp, 'shap_out': out, 'band': band, 'sup': sup},
          open(OUT + 'results.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o), ensure_ascii=False, indent=1)
