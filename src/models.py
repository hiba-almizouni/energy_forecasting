"""
Les quatre modèles demandés (ARIMA, Prophet, XGBoost, LSTM) + une référence
(naïf saisonnier), avec le MÊME protocole pour tous.

PROTOCOLE
---------
Les données sont découpées dans l'ordre chronologique : 70 % train, 15 % validation,
15 % test.

* Choix des hyperparamètres : on ajuste chaque candidat sur le TRAIN et on le note
  sur la VALIDATION (RMSE). Le TEST n'est jamais utilisé à ce stade.
* Modèle final : on réajuste les paramètres retenus sur TRAIN + VALIDATION, puis on
  prédit le TEST une seule fois.
* Prévision "rolling origin" : à chaque origine t, le modèle prédit les h pas suivants
  (h = horizon du jeu de données : 12 mois ou 24 h). L'origine avance ensuite de h pas.
  ARIMA / XGBoost / LSTM utilisent les valeurs observées jusqu'à l'origine (sans
  ré-estimer leurs paramètres). Prophet est un modèle de la date : il n'utilise pas
  les observations récentes, c'est sa nature.
* Les variables explicatives (météo) sont supposées connues sur l'horizon
  ("prévision parfaite" de la météo). C'est une hypothèse classique, à discuter.
"""
from __future__ import annotations

import itertools
import logging
import time
import warnings

import numpy as np
import pandas as pd

import config
from . import metrics as M

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

ROLLS = {"MS": [3, 12], "h": [24, 168], "D": [7, 28], "30min": [48, 336], "15min": [96, 672]}


def _post(ds, pred, lo=None, up=None):
    """Une production d'énergie ne peut pas être négative."""
    if ds.cfg.get("clip_negative", False):
        pred = np.clip(pred, 0, None)
        if lo is not None:
            lo = np.clip(lo, 0, None)
            up = np.clip(up, 0, None)
    return pred, lo, up


class BaseModel:
    name = "base"

    # ---- à définir dans chaque modèle ----
    def grid(self, ds, use_exog):
        raise NotImplementedError

    def _fit(self, ds, fit_end, params, use_exog, val_range=None):
        raise NotImplementedError

    def _predict(self, state, ds, start, end):
        raise NotImplementedError

    def n_adj_params(self, state, ds, use_exog):
        return 1

    def finalize_params(self, params, state):
        return dict(params)

    def describe(self, params, ds=None):
        return str(params)

    # ---- commun ----
    def select(self, ds, use_exog):
        va0, va1 = ds.val_slice
        y_val = ds.y_obs.values[va0:va1]
        best, rows, t0 = None, [], time.time()
        cands = list(self.grid(ds, use_exog))
        for i, params in enumerate(cands, 1):
            t1 = time.time()
            try:
                st = self._fit(ds, ds.n_train, params, use_exog, val_range=(va0, va1))
                pred, _, _ = self._predict(st, ds, va0, va1)
                pred, _, _ = _post(ds, pred)
                score = M.rmse(y_val, pred)
            except Exception as e:  # un candidat qui échoue ne doit pas tout arrêter
                rows.append(dict(params=self.describe(params, ds), val_RMSE=np.nan,
                                 temps_s=time.time() - t1, erreur=str(e)[:100]))
                continue
            rows.append(dict(params=self.describe(self.finalize_params(params, st), ds),
                             val_RMSE=score, temps_s=time.time() - t1, erreur=""))
            if np.isfinite(score) and (best is None or score < best["score"]):
                best = dict(score=score, params=dict(params), state=st, pred=pred)
            print(f"      [{self.name}] candidat {i}/{len(cands)}  RMSE val = {score:.4f}", flush=True)
        if best is None:
            raise RuntimeError(f"{self.name} : aucun candidat n'a pu être ajusté.")
        params = self.finalize_params(best["params"], best["state"])
        resid = y_val - best["pred"]
        pi_q = float(np.nanquantile(np.abs(resid), 1 - config.ALPHA_PI))
        return dict(params=params, val_pred=best["pred"], val_RMSE=best["score"], log=rows,
                    select_time=time.time() - t0, pi_q=pi_q, state_val=best["state"],
                    p_adj=self.n_adj_params(best["state"], ds, use_exog),
                    label=self.describe(params, ds))

    def final(self, ds, use_exog, params, pi_q):
        te0, te1 = ds.test_slice
        t0 = time.time()
        st = self._fit(ds, te0, params, use_exog, val_range=None)   # train + validation
        fit_time = time.time() - t0
        t0 = time.time()
        pred, lo, up = self._predict(st, ds, te0, te1)
        pred_time = time.time() - t0
        pi_type = "modèle"
        if lo is None:
            lo, up = pred - pi_q, pred + pi_q
            pi_type = "empirique (résidus de validation)"
        pred, lo, up = _post(ds, pred, lo, up)
        return dict(pred=pred, lo=lo, up=up, fit_time=fit_time, predict_time=pred_time,
                    p_adj=self.n_adj_params(st, ds, use_exog), state=st, pi_type=pi_type)


# =============================================================================
# 1) ARIMA / SARIMA
# =============================================================================
class ARIMAModel(BaseModel):
    name = "ARIMA"

    def grid(self, ds, use_exog):
        fast = config.FAST_MODE
        if ds.freq == "MS":                      # SARIMA mensuel (saisonnalité 12)
            pq = [0, 1] if fast else [0, 1, 2]
            sq = [0, 1]
            return [dict(order=(p, 1, q), seasonal=(P, 1, Q, 12), K=0)
                    for p, q, P, Q in itertools.product(pq, pq, sq, sq)]
        kmax = max(1, ds.period // 2)           # sous-journalier : ARIMA + termes de Fourier
        Ks = [min(3, kmax)] if fast else sorted({min(3, kmax), min(6, kmax)})
        orders = [(1, 0, 0), (1, 0, 1)] if fast else [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1)]
        return [dict(order=o, seasonal=(0, 0, 0, 0), K=K) for o in orders for K in Ks]

    def describe(self, p, ds=None):
        o, s = p["order"], p["seasonal"]
        if s[3]:
            return f"SARIMA{o}x{s[:3]}[{s[3]}]"
        return f"ARIMA{o} + Fourier(K={p['K']})"

    def _exog(self, ds, use_exog, K, fit_end):
        cols = []
        if K > 0:
            t = np.arange(ds.n)
            for k in range(1, K + 1):
                cols += [np.sin(2 * np.pi * k * t / ds.period), np.cos(2 * np.pi * k * t / ds.period)]
        if use_exog and ds.has_exog:
            mu = ds.exog.iloc[:fit_end].mean()
            sd = ds.exog.iloc[:fit_end].std().replace(0, 1)
            cols += list(((ds.exog - mu) / sd).values.T)
        return np.column_stack(cols) if cols else None

    def _fit(self, ds, fit_end, params, use_exog, val_range=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        sd = float(np.nanstd(ds.y_obs.values[:fit_end])) or 1.0
        y = ds.y_obs.values / sd
        X = self._exog(ds, use_exog, params["K"], fit_end)
        d, D = params["order"][1], params["seasonal"][1]
        mod = SARIMAX(y[:fit_end], exog=None if X is None else X[:fit_end],
                      order=params["order"], seasonal_order=params["seasonal"],
                      trend="c" if (d == 0 and D == 0) else "n",
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=100)
        return dict(res=res, sd=sd, X=X, y=y, params=params)

    def _predict(self, st, ds, start, end):
        h, res, y, X, sd = ds.horizon, st["res"], st["y"], st["X"], st["sd"]
        n = end - start
        pred, lo, up = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for s in range(start, end, h):
            e = min(s + h, end)
            r = res.apply(y[:s], exog=None if X is None else X[:s], refit=False)
            fc = r.get_forecast(steps=e - s, exog=None if X is None else X[s:e])
            ci = np.asarray(fc.conf_int(alpha=config.ALPHA_PI))
            pred[s - start:e - start] = np.asarray(fc.predicted_mean) * sd
            lo[s - start:e - start] = ci[:, 0] * sd
            up[s - start:e - start] = ci[:, 1] * sd
        return pred, lo, up

    def n_adj_params(self, st, ds, use_exog):
        return len([n for n in st["res"].param_names if n != "sigma2"])


# =============================================================================
# 2) Prophet
# =============================================================================
class ProphetModel(BaseModel):
    name = "Prophet"

    def grid(self, ds, use_exog):
        fast = config.FAST_MODE
        cps = [0.05] if fast else [0.01, 0.05, 0.5]
        modes = ["additive"] if fast else ["additive", "multiplicative"]
        return [dict(cps=c, mode=m) for c in cps for m in modes]

    def describe(self, p, ds=None):
        return f"mode={p['mode']}, changepoint_prior={p['cps']}"

    def _regs(self, ds, use_exog):
        return list(ds.exog.columns) if (use_exog and ds.has_exog) else []

    def _fit(self, ds, fit_end, params, use_exog, val_range=None):
        from prophet import Prophet
        regs = self._regs(ds, use_exog)
        df = pd.DataFrame({"ds": ds.y.index[:fit_end], "y": ds.y_obs.values[:fit_end]})
        for c in regs:
            df[c] = ds.exog[c].values[:fit_end]
        span = (ds.y.index[fit_end - 1] - ds.y.index[0]).days
        sub_daily = ds.freq in ("h", "30min", "15min")
        m = Prophet(growth="linear", yearly_seasonality=bool(span >= 2 * 365),
                    weekly_seasonality=bool(ds.cfg.get("prophet_weekly") and ds.freq != "MS" and span >= 14),
                    daily_seasonality=sub_daily, seasonality_mode=params["mode"],
                    changepoint_prior_scale=params["cps"], interval_width=1 - config.ALPHA_PI,
                    uncertainty_samples=300)
        for c in regs:
            m.add_regressor(c, mode=params["mode"])
        np.random.seed(config.SEED)
        m.fit(df.dropna(subset=["y"]))
        return dict(m=m, regs=regs)

    def _predict(self, st, ds, start, end):
        fut = pd.DataFrame({"ds": ds.y.index[start:end]})
        for c in st["regs"]:
            fut[c] = ds.exog[c].values[start:end]
        np.random.seed(config.SEED)
        fc = st["m"].predict(fut)
        return fc["yhat"].values, fc["yhat_lower"].values, fc["yhat_upper"].values

    def n_adj_params(self, st, ds, use_exog):
        m = st["m"]
        k = 1 + sum(2 * s["fourier_order"] for s in m.seasonalities.values())
        return int(k + len(st["regs"]))


# =============================================================================
# 3) XGBoost
# =============================================================================
class XGBModel(BaseModel):
    name = "XGBoost"

    def grid(self, ds, use_exog):
        fast = config.FAST_MODE
        depths = [4] if fast else [3, 5, 7]
        lrs = [0.1] if fast else [0.03, 0.1]
        return [dict(max_depth=d, learning_rate=l, target=t)
                for d in depths for l in lrs for t in ("level", "delta")]

    def describe(self, p, ds=None):
        s = f"profondeur={p['max_depth']}, lr={p['learning_rate']}, cible={p['target']}"
        if "n_estimators" in p:
            s += f", arbres={p['n_estimators']}"
        return s

    def _features(self, ds, use_exog):
        h, P, y = ds.horizon, ds.period, ds.y
        lags = set(range(h, h + P)) | {k * P for k in (1, 2, 3) if k * P >= h}
        if ds.freq == "h" and 168 >= h:
            lags.add(168)
        lags = sorted(lags)
        F = {f"lag_{L}": y.shift(L) for L in lags}
        ys = y.shift(h)
        for w in ROLLS[ds.freq]:
            F[f"moy_{w}"] = ys.rolling(w, min_periods=w).mean()
            F[f"std_{w}"] = ys.rolling(w, min_periods=w).std()
        F = pd.DataFrame(F, index=y.index)
        F = pd.concat([F, ds.cal_raw], axis=1)
        if use_exog and ds.has_exog:
            F = pd.concat([F, ds.exog.add_prefix("x_")], axis=1)
        first = max(max(lags), max(ROLLS[ds.freq]) + h)
        return F, ys.values, first

    def _fit(self, ds, fit_end, params, use_exog, val_range=None):
        import xgboost as xgb
        F, base, first = self._features(ds, use_exog)
        delta = params["target"] == "delta"
        tgt = ds.y.values - base if delta else ds.y.values
        idx = np.arange(first, fit_end)
        idx = idx[ds.valid[idx]]
        kw = dict(max_depth=params["max_depth"], learning_rate=params["learning_rate"],
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  random_state=config.SEED, n_jobs=-1)
        if val_range is not None:                       # phase de choix : arrêt précoce sur la validation
            v0, v1 = val_range
            vi = np.arange(v0, v1)
            vi = vi[ds.valid[vi]]
            model = xgb.XGBRegressor(n_estimators=config.XGB_MAX_ROUNDS // (4 if config.FAST_MODE else 1),
                                     early_stopping_rounds=config.XGB_EARLY_STOP, **kw)
            model.fit(F.iloc[idx], tgt[idx], eval_set=[(F.iloc[vi], tgt[vi])], verbose=False)
            best_n = int(model.best_iteration + 1)
        else:                                           # modèle final : nb d'arbres déjà choisi
            model = xgb.XGBRegressor(n_estimators=int(params["n_estimators"]), **kw)
            model.fit(F.iloc[idx], tgt[idx], verbose=False)
            best_n = int(params["n_estimators"])
        imp = pd.Series(model.feature_importances_, index=F.columns).sort_values(ascending=False)
        return dict(model=model, delta=delta, best_n=best_n, n_feat=F.shape[1], importance=imp,
                    use_exog=use_exog)

    def finalize_params(self, params, state):
        p = dict(params)
        p["n_estimators"] = state["best_n"]
        return p

    def _predict(self, st, ds, start, end):
        F, base, _ = self._features(ds, st["use_exog"])
        out = st["model"].predict(F.iloc[start:end])
        if st["delta"]:
            out = out + base[start:end]
        return out, None, None

    def n_adj_params(self, st, ds, use_exog):
        return int(st["n_feat"])


# =============================================================================
# 4) LSTM
# =============================================================================
class LSTMModel(BaseModel):
    name = "LSTM"

    def grid(self, ds, use_exog):
        fast = config.FAST_MODE
        units = [32] if fast else [32, 64]
        return [dict(units=u, target=t) for u in units for t in ("level", "delta")]

    def describe(self, p, ds=None):
        s = f"unités={p['units']}, cible={p['target']}"
        if ds is not None:
            s += f", fenêtre={config.LSTM_SEQ_LEN[ds.freq]}"
        if "epochs" in p:
            s += f", epochs={p['epochs']}"
        return s

    def _prepare(self, ds, use_exog):
        h = ds.horizon
        n_tr = ds.n_train
        mu_y = float(np.nanmean(ds.y_obs.values[:n_tr]))
        sd_y = float(np.nanstd(ds.y_obs.values[:n_tr])) or 1.0
        ys = ((ds.y.values - mu_y) / sd_y)
        cols = [ys[:, None], ds.cal.values]
        if use_exog and ds.has_exog:
            ex = ds.exog.values
            mu, sd = ex[:n_tr].mean(0), ex[:n_tr].std(0)
            sd[sd == 0] = 1
            cols.append((ex - mu) / sd)
        S = np.hstack(cols).astype("float32")
        pad = np.zeros((h, S.shape[1]), dtype="float32")
        S = np.vstack([S, pad])
        ys = np.concatenate([ys, np.zeros(h)]).astype("float32")
        valid = np.concatenate([ds.valid, np.zeros(h, bool)])
        return S, ys, valid, mu_y, sd_y

    def _windows(self, S, ys, valid, L, h):
        from numpy.lib.stride_tricks import sliding_window_view as swv
        Xs = np.transpose(swv(S, L, axis=0), (0, 2, 1))          # (n-L+1, L, F)
        T = swv(ys, h)                                             # (n-h+1, h)
        Fu = np.transpose(swv(S[:, 1:], h, axis=0), (0, 2, 1))    # (n-h+1, h, Fc)
        Fu = Fu.reshape(Fu.shape[0], -1)
        V = swv(valid, h).all(axis=1)
        return Xs, T, Fu, V

    def _batch(self, W, ys, origins, L, delta):
        Xs, T, Fu, V = W
        o = np.asarray(origins)
        base = ys[o][:, None] if delta else 0.0
        return Xs[o - L + 1], Fu[o + 1], T[o + 1] - base, V[o + 1]

    def _fit(self, ds, fit_end, params, use_exog, val_range=None):
        import keras
        from keras import layers
        keras.utils.set_random_seed(config.SEED)
        keras.backend.clear_session()
        h, L = ds.horizon, config.LSTM_SEQ_LEN[ds.freq]
        S, ys, valid, mu_y, sd_y = self._prepare(ds, use_exog)
        W = self._windows(S, ys, valid, L, h)
        delta = params["target"] == "delta"
        F, Fc = S.shape[1], S.shape[1] - 1
        o_tr = np.arange(L - 1, fit_end - h)
        Xtr, Ftr, Ytr, Vtr = self._batch(W, ys, o_tr, L, delta)
        Xtr, Ftr, Ytr = Xtr[Vtr], Ftr[Vtr], Ytr[Vtr]

        seq_in = keras.Input((L, F))
        fut_in = keras.Input((h * Fc,))
        x = layers.LSTM(params["units"])(seq_in)
        x = layers.Dropout(0.1)(x)
        z = layers.Concatenate()([x, fut_in])
        z = layers.Dense(params["units"], activation="relu")(z)
        out = layers.Dense(h)(z)
        model = keras.Model([seq_in, fut_in], out)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        max_ep = 15 if config.FAST_MODE else config.LSTM_MAX_EPOCHS
        if val_range is not None:
            v0, v1 = val_range
            o_v = np.arange(v0 - 1, v1 - h)
            Xv, Fv, Yv, Vv = self._batch(W, ys, o_v, L, delta)
            hist = model.fit([Xtr, Ftr], Ytr, validation_data=([Xv[Vv], Fv[Vv]], Yv[Vv]),
                             epochs=max_ep, batch_size=64, verbose=0,
                             callbacks=[keras.callbacks.EarlyStopping(
                                 monitor="val_loss", patience=config.LSTM_PATIENCE if not config.FAST_MODE else 4,
                                 restore_best_weights=True)])
            val_loss = hist.history["val_loss"]
            best_ep = int(np.argmin(val_loss) + 1)
        else:
            best_ep = int(params["epochs"])
            hist = model.fit([Xtr, Ftr], Ytr, epochs=best_ep, batch_size=64, verbose=0)
        return dict(model=model, W=W, ys=ys, L=L, delta=delta, mu=mu_y, sd=sd_y, F=F,
                    best_epochs=best_ep, history=hist.history, n_samples=int(len(Xtr)))

    def finalize_params(self, params, state):
        p = dict(params)
        p["epochs"] = state["best_epochs"]
        return p

    def _predict(self, st, ds, start, end):
        h, L = ds.horizon, st["L"]
        origins = np.arange(start - 1, end - 1, h)
        X, Fu, _, _ = self._batch(st["W"], st["ys"], origins, L, st["delta"])
        out = st["model"].predict([X, Fu], verbose=0, batch_size=256)
        if st["delta"]:
            out = out + st["ys"][origins][:, None]
        out = out * st["sd"] + st["mu"]
        return out.reshape(-1)[: end - start].astype(float), None, None

    def n_adj_params(self, st, ds, use_exog):
        return int(st["F"])


# =============================================================================
# Référence : naïf saisonnier
# =============================================================================
def seasonal_naive(ds, start, end):
    h, P = ds.horizon, ds.period
    lag = P * int(np.ceil(h / P))
    idx = np.arange(start, end) - lag
    return ds.y.values[idx]


def make_models():
    return [ARIMAModel(), ProphetModel(), XGBModel(), LSTMModel()]
