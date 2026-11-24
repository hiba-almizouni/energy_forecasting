"""Pipeline complet pour un jeu de données : EDA -> modèles -> évaluation -> figures."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from . import data_loader as dl
from . import eda
from . import metrics as M
from . import models as MD

COLORS = {"ARIMA": "#2b6cb0", "Prophet": "#38a169", "XGBoost": "#dd6b20",
          "LSTM": "#805ad5", "Naïf saisonnier": "#718096"}


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------------
def error_analysis(ds, y, pred, name):
    te0, te1 = ds.test_slice
    idx = ds.y.index[te0:te1]
    h = ds.horizon
    e = y - pred
    ok = np.isfinite(e)
    sub_daily = ds.freq in ("h", "30min", "15min")
    key = idx.hour if sub_daily else idx.month
    by_key = pd.Series(np.abs(e), index=key).groupby(level=0).mean()
    step = (np.arange(len(e)) % h) + 1
    by_step = pd.Series(np.abs(e), index=step).groupby(level=0).mean()
    worst = pd.Series(np.abs(e), index=idx).dropna().sort_values(ascending=False).head(5).index
    return dict(
        biais=float(np.nanmean(e)), erreur_max=float(np.nanmax(np.abs(e))),
        mae_par_groupe={int(k): float(v) for k, v in by_key.items()},
        groupe="heure" if sub_daily else "mois",
        mae_par_pas={int(k): float(v) for k, v in by_step.items()},
        pires=[dict(date=str(d), reel=float(y[idx.get_loc(d)]), prevu=float(pred[idx.get_loc(d)])) for d in worst],
        part_hors_intervalle=None,
    )


def run_variant(ds, vname, use_exog, out: Path):
    te0, te1 = ds.test_slice
    va0, va1 = ds.val_slice
    y_va, y_te = ds.y_obs.values[va0:va1], ds.y_obs.values[te0:te1]
    h, P = ds.horizon, ds.period
    idx_te = ds.y.index[te0:te1]
    print(f"\n  >>> Variante : {vname}  (variables explicatives : {'oui' if use_exog else 'non'})", flush=True)

    R = dict(use_exog=use_exog, models={}, figures=[])
    store = {}
    for m in MD.make_models():
        print(f"    - {m.name} : choix des hyperparamètres sur la validation...", flush=True)
        try:
            sel = m.select(ds, use_exog)
            print(f"      meilleur : {sel['label']} (RMSE val = {sel['val_RMSE']:.4f}). Ajustement final...", flush=True)
            fin = m.final(ds, use_exog, sel["params"], sel["pi_q"])
        except Exception as e:
            print(f"      !! {m.name} a échoué : {e}")
            R["models"][m.name] = dict(erreur=str(e))
            continue
        store[m.name] = (sel, fin)
        mv = M.full_report(y_va, sel["val_pred"], sel["p_adj"], h, P)
        mt = M.full_report(y_te, fin["pred"], fin["p_adj"], h, P)
        pi = M.interval_scores(y_te, fin["lo"], fin["up"])
        pi["type"] = fin["pi_type"]
        R["models"][m.name] = dict(
            label=sel["label"], params=sel["params"], val=mv, test=mt, pi=pi,
            temps=dict(choix_s=sel["select_time"], ajustement_s=fin["fit_time"],
                       prediction_s=fin["predict_time"]),
            grille=sel["log"])

    # référence naïve
    nv, nt = MD.seasonal_naive(ds, va0, va1), MD.seasonal_naive(ds, te0, te1)
    nv, _, _ = MD._post(ds, nv)
    nt, _, _ = MD._post(ds, nt)
    R["naif"] = dict(label="valeur de la même saison (décalage %d)" % (P * int(np.ceil(h / P))),
                     val=M.full_report(y_va, nv, 0, h, P), test=M.full_report(y_te, nt, 0, h, P))

    names = list(store.keys())
    if not names:
        raise RuntimeError("Aucun modèle n'a pu être exécuté.")
    R["best_val"] = min(names, key=lambda k: R["models"][k]["val"]["RMSE"])
    R["best_test"] = min(names, key=lambda k: R["models"][k]["test"]["RMSE"])

    # --- tests de Diebold-Mariano (test set) ---
    preds = {k: store[k][1]["pred"] for k in names}
    preds["Naïf saisonnier"] = nt
    allnames = list(preds.keys())
    R["dm"] = {a: {b: M.dm_test(y_te, preds[a], preds[b], h) for b in allnames if b != a} for a in allnames}

    # --- analyse des erreurs ---
    R["erreurs"] = {}
    for k in names:
        ea = error_analysis(ds, y_te, preds[k], k)
        ea["part_hors_intervalle"] = 1 - (R["models"][k]["pi"]["coverage"] or 0)
        R["erreurs"][k] = ea

    # --- sauvegardes tabulaires ---
    df = pd.DataFrame({"date": idx_te, "reel": y_te})
    for k in names:
        df[k] = store[k][1]["pred"]
        df[k + "_bas"] = store[k][1]["lo"]
        df[k + "_haut"] = store[k][1]["up"]
    df["Naïf saisonnier"] = nt
    df.to_csv(out / f"predictions_test_{vname}.csv", index=False)
    rows = []
    for k in names:
        rows.append(dict(modele=k, **{c: R["models"][k]["test"][c] for c in
                                      ("MAE", "RMSE", "R2", "R2_adj", "p_pente", "p_MZ", "nRMSE", "biais", "p_LjungBox")}))
    pd.DataFrame(rows).to_csv(out / f"metriques_test_{vname}.csv", index=False)

    # --- figures ---
    fd = out / "figures"
    # (a) prévisions vs réel
    sub_daily = ds.freq in ("h", "30min", "15min")
    zoom = min(7 * P, len(idx_te)) if sub_daily else len(idx_te)
    nrow = 2 if sub_daily else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(11, 3.6 * nrow))
    axes = np.atleast_1d(axes)
    for a, (n_pts, ttl) in zip(axes, [(len(idx_te), "Ensemble du jeu de test"), (zoom, f"Zoom : {zoom} premiers pas")][:nrow]):
        a.plot(idx_te[:n_pts], y_te[:n_pts], color="black", lw=1.6, label="Réel")
        for k in names:
            a.plot(idx_te[:n_pts], preds[k][:n_pts], color=COLORS[k], lw=1.0, alpha=.9, label=k)
        a.set_title(ttl)
        a.set_ylabel(ds.unit)
    axes[0].legend(ncol=5, fontsize=8, loc="upper left")
    fig.suptitle(f"{ds.title} - valeurs réelles et prédites ({vname})", y=1.0)
    R["figures"].append((_save(fig, fd / f"prev_test_{vname}.png"),
                         f"Valeurs réelles et prédites sur le jeu de test ({vname})."))
    # (b) intervalles de prédiction
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.8 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for a, k in zip(axes, names):
        a.fill_between(idx_te[:zoom], store[k][1]["lo"][:zoom], store[k][1]["up"][:zoom],
                       color=COLORS[k], alpha=.2, label="Intervalle 95 %")
        a.plot(idx_te[:zoom], y_te[:zoom], color="black", lw=1.2, label="Réel")
        a.plot(idx_te[:zoom], preds[k][:zoom], color=COLORS[k], lw=1.2, label=k)
        a.set_ylabel(ds.unit)
        a.set_title(f"{k} - couverture {R['models'][k]['pi']['coverage']:.0%} ({R['models'][k]['pi']['type']})", fontsize=9)
        a.legend(loc="upper left", fontsize=7, ncol=3)
    R["figures"].append((_save(fig, fd / f"intervalles_{vname}.png"),
                         f"Intervalles de prédiction à 95 % ({vname})."))
    # (c) nuages réel vs prévu
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.4))
    axes = np.atleast_1d(axes)
    lo_, hi_ = np.nanmin(y_te), np.nanmax(y_te)
    for a, k in zip(axes, names):
        a.scatter(y_te, preds[k], s=6, alpha=.35, color=COLORS[k])
        a.plot([lo_, hi_], [lo_, hi_], "k--", lw=1)
        a.set_title(f"{k}  (R² = {R['models'][k]['test']['R2']:.3f})", fontsize=9)
        a.set_xlabel("Réel")
    axes[0].set_ylabel("Prédit")
    R["figures"].append((_save(fig, fd / f"nuage_{vname}.png"), f"Prédit en fonction du réel ({vname})."))
    # (d) résidus
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k in names:
        axes[0].hist((y_te - preds[k])[np.isfinite(y_te - preds[k])], bins=40, alpha=.45,
                     color=COLORS[k], label=k, density=True)
    axes[0].set_title("Distribution des erreurs (réel - prédit)")
    axes[0].legend(fontsize=8)
    for k in names:
        st = R["erreurs"][k]["mae_par_pas"]
        axes[1].plot(list(st.keys()), list(st.values()), marker="o", ms=3, color=COLORS[k], label=k)
    axes[1].set_title("MAE selon le pas de prévision dans l'horizon")
    axes[1].set_xlabel("Pas (1 = premier pas après l'origine)")
    axes[1].set_ylabel(f"MAE ({ds.unit})")
    R["figures"].append((_save(fig, fd / f"residus_{vname}.png"), f"Distribution des erreurs et erreur selon le pas ({vname})."))
    # (e) erreur par heure / mois
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for k in names:
        g = R["erreurs"][k]["mae_par_groupe"]
        ax.plot(list(g.keys()), list(g.values()), marker="o", ms=3, color=COLORS[k], label=k)
    ax.set_xlabel(R["erreurs"][names[0]]["groupe"].capitalize())
    ax.set_ylabel(f"MAE ({ds.unit})")
    ax.set_title("Erreur absolue moyenne par " + R["erreurs"][names[0]]["groupe"])
    ax.legend(fontsize=8)
    R["figures"].append((_save(fig, fd / f"erreur_groupe_{vname}.png"), f"MAE selon l'heure ou le mois ({vname})."))
    # (f) barres de métriques
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    allm = names + ["Naïf saisonnier"]
    for a, met in zip(axes, ["RMSE", "MAE"]):
        vals = [R["models"][k]["test"][met] for k in names] + [R["naif"]["test"][met]]
        a.bar(allm, vals, color=[COLORS[k] for k in allm])
        a.set_title(f"{met} sur le jeu de test")
        a.tick_params(axis="x", rotation=25)
    R["figures"].append((_save(fig, fd / f"barres_{vname}.png"), f"RMSE et MAE sur le jeu de test ({vname})."))
    # (g) courbe d'apprentissage LSTM
    if "LSTM" in store:
        hs = store["LSTM"][0]["state_val"]["history"]
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.plot(hs["loss"], label="Train")
        if "val_loss" in hs:
            ax.plot(hs["val_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (série standardisée)")
        ax.set_title("LSTM - courbe d'apprentissage")
        ax.legend()
        R["figures"].append((_save(fig, fd / f"lstm_loss_{vname}.png"), f"Courbe d'apprentissage du LSTM ({vname})."))
    # (h) importance XGBoost
    if "XGBoost" in store:
        imp = store["XGBoost"][1]["state"]["importance"].head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.barh(imp.index, imp.values, color="#dd6b20")
        ax.set_title("XGBoost - importance des variables (top 15)")
        R["figures"].append((_save(fig, fd / f"xgb_importance_{vname}.png"), f"Importance des variables de XGBoost ({vname})."))
        R["xgb_top"] = {k: float(v) for k, v in store["XGBoost"][1]["state"]["importance"].head(8).items()}
    # (i) quelques infos sur le modèle ARIMA final
    if "ARIMA" in store:
        res = store["ARIMA"][1]["state"]["res"]
        R["arima_aic"] = float(res.aic)
    return R


def run_dataset(key: str, outroot: Path) -> dict:
    t0 = time.time()
    print(f"\n=== Jeu de données : {key} ===")
    ds = dl.load_dataset(key)
    out = Path(outroot) / key
    (out / "figures").mkdir(parents=True, exist_ok=True)
    print("  EDA...", flush=True)
    eda_res = eda.run_eda(ds, out)
    ds.raw.to_csv(out / "donnees_nettoyees.csv")
    variants = [("avec_exog", True), ("sans_exog", False)] if ds.has_exog else [("univarie", False)]
    variants = sorted(variants, key=lambda v: v[1])          # sans_exog d'abord
    results = {}
    for vname, use_exog in variants:
        results[vname] = run_variant(ds, vname, use_exog, out)
    payload = dict(
        key=key, title=ds.title, unit=ds.unit, freq=ds.freq, period=ds.period, horizon=ds.horizon,
        info=ds.info, eda=eda_res, variants=results, has_exog=ds.has_exog,
        exog_cols=list(ds.exog.columns), duree_s=time.time() - t0,
        fast_mode=config.FAST_MODE)
    with open(out / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(to_py(payload), f, ensure_ascii=False, indent=1)
    print(f"  Terminé en {payload['duree_s'] / 60:.1f} min -> {out}")
    return payload
