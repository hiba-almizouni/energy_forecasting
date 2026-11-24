"""
Métriques d'erreur et tests statistiques.

MAE, RMSE, R², R² ajusté, p-values (régression de Mincer-Zarnowitz),
test de Ljung-Box sur les résidus, test de Diebold-Mariano entre modèles,
couverture des intervalles de prédiction.
"""
from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def _clean(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m], m


def mae(y, yhat):
    y, yhat, _ = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    y, yhat, _ = _clean(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2(y, yhat):
    y, yhat, _ = _clean(y, yhat)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / sst) if sst > 0 else float("nan")


def adj_r2(y, yhat, p):
    """R² ajusté = 1 - (1 - R²)(n - 1)/(n - p - 1), p = nb de variables explicatives du modèle."""
    y, yhat, _ = _clean(y, yhat)
    n = len(y)
    if n - p - 1 <= 0:
        return float("nan")
    return float(1 - (1 - r2(y, yhat)) * (n - 1) / (n - p - 1))


def mincer_zarnowitz(y, yhat, maxlags: int = 1):
    """
    Régression y = a + b * yhat (erreurs standard robustes HAC / Newey-West).
      - p_slope : test de H0 "b = 0"  (la prévision n'apporte aucune information)
      - p_mz    : test de H0 "a = 0 et b = 1" (prévision sans biais et efficace)
    """
    y, yhat, _ = _clean(y, yhat)
    out = dict(intercept=np.nan, slope=np.nan, p_slope=np.nan, p_mz=np.nan)
    if len(y) < 10 or np.std(yhat) == 0:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, sm.add_constant(yhat)).fit(
                cov_type="HAC", cov_kwds={"maxlags": int(max(1, maxlags))})
            wald = res.wald_test("const = 0, x1 = 1", scalar=True)
        out.update(intercept=float(res.params[0]), slope=float(res.params[1]),
                   p_slope=float(res.pvalues[1]), p_mz=float(wald.pvalue))
    except Exception:
        pass
    return out


def ljung_box_p(resid, lags: int):
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 * lags:
        lags = max(1, len(r) // 5)
    try:
        return float(acorr_ljungbox(r, lags=[lags], return_df=True)["lb_pvalue"].iloc[0])
    except Exception:
        return float("nan")


def dm_test(y, yhat1, yhat2, h: int = 1):
    """
    Test de Diebold-Mariano (perte quadratique), correction de Harvey-Leybourne-Newbold.
    H0 : les deux modèles ont la même précision.
    Statistique > 0 : le modèle 1 fait de plus grosses erreurs que le modèle 2.
    """
    y = np.asarray(y, float)
    e1 = y - np.asarray(yhat1, float)
    e2 = y - np.asarray(yhat2, float)
    m = np.isfinite(e1) & np.isfinite(e2)
    d = (e1[m] ** 2) - (e2[m] ** 2)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return dict(stat=np.nan, p=np.nan)
    dbar = d.mean()
    dc = d - dbar
    lags = max(h - 1, 0)
    gamma0 = np.sum(dc * dc) / n
    var = gamma0
    for k in range(1, lags + 1):
        gk = np.sum(dc[k:] * dc[:-k]) / n
        var += 2 * (1 - k / (lags + 1)) * gk      # noyau de Bartlett
    var = var / n
    if var <= 0:
        return dict(stat=np.nan, p=np.nan)
    stat = dbar / np.sqrt(var)
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    stat_c = stat * corr
    p = 2 * (1 - stats.t.cdf(abs(stat_c), df=n - 1))
    return dict(stat=float(stat_c), p=float(p))


def interval_scores(y, lo, up):
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    up = np.asarray(up, float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(up)
    if m.sum() == 0:
        return dict(coverage=np.nan, width=np.nan)
    cov = np.mean((y[m] >= lo[m]) & (y[m] <= up[m]))
    return dict(coverage=float(cov), width=float(np.mean(up[m] - lo[m])))


def full_report(y, yhat, p_params: int, h: int, period: int):
    """Toutes les métriques d'un modèle sur un jeu d'évaluation."""
    y_, yh_, m = _clean(y, yhat)
    mz = mincer_zarnowitz(y_, yh_, maxlags=max(1, h))
    resid = y_ - yh_
    mean_y = float(np.mean(y_)) if len(y_) else np.nan
    return dict(
        n=int(len(y_)), MAE=mae(y_, yh_), RMSE=rmse(y_, yh_), R2=r2(y_, yh_),
        R2_adj=adj_r2(y_, yh_, p_params), p_params=int(p_params),
        nRMSE=rmse(y_, yh_) / mean_y if mean_y else np.nan,
        biais=float(np.mean(resid)) if len(resid) else np.nan,
        p_pente=mz["p_slope"], p_MZ=mz["p_mz"],
        pente=mz["slope"], ordonnee=mz["intercept"],
        p_LjungBox=ljung_box_p(resid, lags=min(2 * period, 48)),
    )
