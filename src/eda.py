"""Analyse exploratoire des données (EDA) : tableaux, figures, tests."""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
C_TRAIN, C_VAL, C_TEST = "#2b6cb0", "#dd8a1b", "#c53030"


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _adf(x):
    x = pd.Series(x).dropna()
    try:
        r = adfuller(x, autolag="AIC")
        return dict(stat=float(r[0]), p=float(r[1]))
    except Exception:
        return dict(stat=np.nan, p=np.nan)


def _kpss(x):
    x = pd.Series(x).dropna()
    try:
        r = kpss(x, regression="c", nlags="auto")
        return dict(stat=float(r[0]), p=float(r[1]))
    except Exception:
        return dict(stat=np.nan, p=np.nan)


def run_eda(ds, outdir: Path) -> dict:
    fig_dir = Path(outdir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y, unit, P = ds.y, ds.unit, ds.period
    sub_daily = ds.freq in ("h", "30min", "15min")
    figs, R = [], {}

    # ---------------- 1. Caractéristiques ----------------
    desc = ds.y_obs.dropna()
    R["stats"] = dict(
        moyenne=float(desc.mean()), ecart_type=float(desc.std()), min=float(desc.min()),
        q25=float(desc.quantile(.25)), mediane=float(desc.median()), q75=float(desc.quantile(.75)),
        max=float(desc.max()), asymetrie=float(stats.skew(desc)),
        kurtosis=float(stats.kurtosis(desc)), part_zeros=float((desc == 0).mean()))

    # ---------------- 2. Série temporelle ----------------
    n_tr, n_va = ds.n_train, ds.n_val
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.index[:n_tr], y.values[:n_tr], color=C_TRAIN, lw=.7 if sub_daily else 1.2, label="Train (70 %)")
    ax.plot(y.index[n_tr:n_tr + n_va], y.values[n_tr:n_tr + n_va], color=C_VAL,
            lw=.7 if sub_daily else 1.2, label="Validation (15 %)")
    ax.plot(y.index[n_tr + n_va:], y.values[n_tr + n_va:], color=C_TEST,
            lw=.7 if sub_daily else 1.2, label="Test (15 %)")
    bad = ~ds.valid
    if bad.any():
        ax.scatter(y.index[bad], y.values[bad], s=3, color="k", alpha=.4, label="Valeurs reconstruites (trous longs)")
    ax.set_title(f"{ds.title} - évolution temporelle et découpage chronologique")
    ax.set_ylabel(unit)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    figs.append((_save(fig, fig_dir / "eda_serie.png"),
                 "Évolution de la production et découpage chronologique 70/15/15."))

    # zoom / profil saisonnier
    if sub_daily:
        z = y.iloc[: 14 * P]
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(z.index, z.values, color=C_TRAIN)
        ax.set_title("Zoom sur 14 jours : cycle journalier")
        ax.set_ylabel(unit)
        figs.append((_save(fig, fig_dir / "eda_zoom.png"), "Zoom sur deux semaines (cycle jour/nuit)."))
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        yy = y.copy()
        tab = pd.DataFrame({"annee": yy.index.year, "mois": yy.index.month, "v": yy.values})
        years = sorted(tab.annee.unique())
        cmap = plt.get_cmap("viridis")
        for i, a in enumerate(years):
            t = tab[tab.annee == a]
            ax.plot(t.mois, t.v, color=cmap(i / max(1, len(years) - 1)), lw=.9, alpha=.85)
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Mois")
        ax.set_ylabel(unit)
        ax.set_title("Profil saisonnier : une courbe par année (clair = récent)")
        figs.append((_save(fig, fig_dir / "eda_saison.png"), "Profil saisonnier annuel : une courbe par année."))

    # ---------------- 3. Décomposition STL ----------------
    try:
        stl = STL(y, period=P, robust=True).fit()
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        for a, (s, t) in zip(axes, [(y, "Observée"), (stl.trend, "Tendance"),
                                    (stl.seasonal, "Saisonnalité"), (stl.resid, "Résidu")]):
            a.plot(s.index, s.values, lw=.7 if sub_daily else 1.1, color="#2d3748")
            a.set_ylabel(t)
        axes[0].set_title("Décomposition STL (robuste)")
        figs.append((_save(fig, fig_dir / "eda_stl.png"), "Décomposition STL : tendance, saisonnalité, résidu."))
        var_r = np.var(stl.resid)
        Ft = max(0.0, 1 - var_r / np.var(stl.trend + stl.resid))
        Fs = max(0.0, 1 - var_r / np.var(stl.seasonal + stl.resid))
        yrs = (stl.trend.index - stl.trend.index[0]).days.values / 365.25
        slope = float(np.polyfit(yrs, stl.trend.values, 1)[0]) if yrs[-1] > 0 else np.nan
        seas_amp = float(stl.seasonal.max() - stl.seasonal.min())
        R["stl"] = dict(force_tendance=float(Ft), force_saison=float(Fs), pente_par_an=slope,
                        amplitude_saison=seas_amp,
                        tendance_debut=float(stl.trend.iloc[:P].mean()), tendance_fin=float(stl.trend.iloc[-P:].mean()))
        # anomalies : résidu STL très éloigné de la médiane
        r = stl.resid
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        z = (r - np.median(r)) / (mad if mad > 0 else 1)
        an = (np.abs(z) > 4) & ds.valid
        R["anomalies"] = dict(n=int(an.sum()), part=float(an.mean()), seuil="|z robuste| > 4",
                              top=[(str(i), float(y[i]), float(z[i])) for i in
                                   z[an].abs().sort_values(ascending=False).head(5).index])
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(y.index, y.values, lw=.6 if sub_daily else 1.0, color="#4a5568")
        ax.scatter(y.index[an.values], y.values[an.values], s=14, color="#c53030", zorder=3,
                   label=f"Points atypiques ({int(an.sum())})")
        ax.set_title("Détection d'anomalies (résidu STL, z robuste > 4)")
        ax.set_ylabel(unit)
        ax.legend(loc="upper left")
        figs.append((_save(fig, fig_dir / "eda_anomalies.png"), "Points atypiques détectés sur le résidu STL."))
    except Exception as e:  # pragma: no cover
        R["stl"] = dict(erreur=str(e))
        R["anomalies"] = dict(n=0, part=0.0, seuil="n/a", top=[])

    # ---------------- 4. Stationnarité ----------------
    R["adf"] = _adf(y)
    R["kpss"] = _kpss(y)
    d1 = y.diff().dropna()
    R["adf_diff1"] = _adf(d1)
    R["adf_diff_saison"] = _adf(y.diff(P).dropna())
    R["kpss_diff1"] = _kpss(d1)

    # ---------------- 5. ACF / PACF ----------------
    nl = 3 * P if not sub_daily else 7 * P
    nl = min(nl, len(y) // 3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_acf(y.dropna(), lags=nl, ax=axes[0], title="ACF")
    plot_pacf(y.dropna(), lags=min(nl, 60), ax=axes[1], title="PACF", method="ywm")
    figs.append((_save(fig, fig_dir / "eda_acf_pacf.png"), "Autocorrélation (ACF) et autocorrélation partielle (PACF)."))

    # ---------------- 6. Distributions ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(desc.values, bins=40, color="#2b6cb0", alpha=.85, density=True)
    axes[0].set_title("Distribution de la production")
    axes[0].set_xlabel(unit)
    key = y.index.hour if sub_daily else y.index.month
    lab = "Heure" if sub_daily else "Mois"
    groups = [y.values[key == k] for k in sorted(set(key))]
    axes[1].boxplot(groups, showfliers=False, tick_labels=[str(k) for k in sorted(set(key))])
    axes[1].set_title(f"Production par {lab.lower()}")
    axes[1].set_xlabel(lab)
    figs.append((_save(fig, fig_dir / "eda_distributions.png"), "Histogramme et boîtes à moustaches par heure ou par mois."))
    if sub_daily:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        mm = pd.Series(y.values, index=y.index).groupby(y.index.month).mean()
        ax.bar(mm.index, mm.values, color="#2b6cb0")
        ax.set_xlabel("Mois")
        ax.set_ylabel(f"Moyenne ({unit})")
        ax.set_title("Production moyenne par mois")
        figs.append((_save(fig, fig_dir / "eda_mois.png"), "Production moyenne par mois présent dans les données."))

    # ---------------- 7. Corrélations ----------------
    num = ds.raw.copy()
    num = num.loc[:, num.std() > 0]
    if num.shape[1] >= 2:
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(1.0 + 0.8 * num.shape[1], 1.0 + 0.7 * num.shape[1]))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.046)
        ax.set_title("Matrice de corrélation (Pearson)")
        figs.append((_save(fig, fig_dir / "eda_corr.png"), "Corrélations entre la production et les variables explicatives."))
        c = corr[ds.target_name].drop(ds.target_name).sort_values(key=np.abs, ascending=False)
        R["corr_target"] = {k: float(v) for k, v in c.items()}
    else:
        R["corr_target"] = {}

    R["figures"] = figs
    return R
