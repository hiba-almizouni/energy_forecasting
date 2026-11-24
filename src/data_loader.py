"""
Chargement, nettoyage et mise en forme des jeux de données.

Un objet `Dataset` contient tout ce dont les modèles ont besoin :
  - y        : série cible complète (trous interpolés) -> sert d'entrée aux modèles
  - y_obs    : série cible avec NaN là où la valeur n'est pas fiable
  - valid    : masque (True = valeur observée ou trou court interpolé)
  - exog     : variables explicatives (météo...), sans NaN
  - cal      : variables de calendrier (sin/cos), déterministes
  - découpage chronologique 70 / 15 / 15
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config

PERIOD_BY_FREQ = {"h": 24, "D": 7, "MS": 12, "30min": 48, "15min": 96}
FREQ_LABEL = {"h": "horaire", "D": "journalière", "MS": "mensuelle",
              "30min": "30 minutes", "15min": "15 minutes"}


@dataclass
class Dataset:
    key: str
    title: str
    unit: str
    freq: str
    period: int
    horizon: int
    target_name: str
    y: pd.Series
    y_obs: pd.Series
    valid: np.ndarray
    exog: pd.DataFrame
    cal: pd.DataFrame
    cal_raw: pd.DataFrame
    raw: pd.DataFrame
    info: dict
    n_train: int
    n_val: int
    n_test: int
    cfg: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def has_exog(self) -> bool:
        return self.exog.shape[1] > 0

    @property
    def val_slice(self):
        return self.n_train, self.n_train + self.n_val

    @property
    def test_slice(self):
        return self.n_train + self.n_val, self.n


# ----------------------------------------------------------------------------
# Outils
# ----------------------------------------------------------------------------
def infer_freq(index: pd.DatetimeIndex) -> str:
    """Déduit la fréquence à partir de l'écart médian entre dates."""
    d = pd.Series(index).diff().dropna().dt.total_seconds().median()
    if d <= 0:
        raise ValueError("Dates dupliquées ou non ordonnées.")
    if abs(d - 900) < 1:
        return "15min"
    if abs(d - 1800) < 1:
        return "30min"
    if abs(d - 3600) < 1:
        return "h"
    if abs(d - 86400) < 1:
        return "D"
    if 27 * 86400 <= d <= 32 * 86400:
        return "MS"
    raise ValueError(f"Fréquence non reconnue (écart médian = {d} s).")


def _short_gap_interp(s: pd.Series, max_gap: int) -> pd.Series:
    """Interpole les trous de longueur <= max_gap ; laisse NaN les trous plus longs."""
    isn = s.isna()
    grp = (isn != isn.shift()).cumsum()
    run = isn.groupby(grp).transform("sum")
    full = s.interpolate(method="time", limit_area="inside")
    return full.mask(isn & (run > max_gap))


def calendar_features(index: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    """Variables de calendrier cycliques (connues à l'avance)."""
    c = pd.DataFrame(index=index)
    if freq in ("h", "30min", "15min"):
        hod = index.hour + index.minute / 60.0
        c["hour_sin"] = np.sin(2 * np.pi * hod / 24)
        c["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    if freq != "MS":
        c["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
        c["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
        c["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
        c["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)
    else:
        c["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
        c["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12)
    return c


def calendar_raw(index: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    """Variables de calendrier entières (pour les arbres)."""
    c = pd.DataFrame(index=index)
    if freq in ("h", "30min", "15min"):
        c["hour"] = index.hour
    if freq != "MS":
        c["dow"] = index.dayofweek
        c["doy"] = index.dayofyear
    c["month"] = index.month
    return c


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def find_file(folder: str, file: str | None) -> Path:
    base = config.DATA_DIR / folder
    if file:
        p = base / file
        if not p.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {p}\n"
                f"Télécharge le jeu de données et place-le dans {base}")
        return p
    cands = sorted([p for p in base.glob("*") if p.suffix.lower() in
                    (".csv", ".xlsx", ".xls", ".parquet", ".txt")])
    if not cands:
        raise FileNotFoundError(
            f"Aucun fichier dans {base}.\n"
            f"Télécharge le jeu de données et place-le dans ce dossier.")
    if len(cands) > 1:
        names = ", ".join(c.name for c in cands)
        print(f"  [info] plusieurs fichiers dans {base} ({names}). "
              f"On utilise {cands[0].name}. Renseigne 'file' dans config.py pour choisir.")
    return cands[0]


# ----------------------------------------------------------------------------
# Chargeurs spécifiques : renvoient un DataFrame indexé par la date
# ----------------------------------------------------------------------------
def load_electric(path: Path, cfg: dict) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    tcol = cfg.get("time_col") or next(
        (c for c in df.columns if c.strip().lower() in ("date", "datetime", "ds")), df.columns[0])
    vcol = cfg.get("target_col") or next(c for c in df.columns if c != tcol)
    out = pd.DataFrame({"production": pd.to_numeric(df[vcol], errors="coerce").values},
                       index=pd.to_datetime(df[tcol]))
    return out, "production"


def load_pvgis(path: Path, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Fichier CSV 'données horaires' exporté par PVGIS (avec en-tête et pied de page)."""
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    start = next((i for i, l in enumerate(lines) if l.lower().startswith("time,")), None)
    if start is None:
        raise ValueError("Format PVGIS non reconnu : ligne d'en-tête 'time,P,...' introuvable.")
    rows = []
    for l in lines[start + 1:]:
        if not l.strip():
            break
        rows.append(l.split(","))
    cols = lines[start].split(",")
    df = pd.DataFrame(rows, columns=cols)
    idx = pd.to_datetime(df["time"], format="%Y%m%d:%H%M").dt.floor("h")
    df = df.drop(columns=["time"]).apply(pd.to_numeric, errors="coerce")
    df.index = idx
    if "P" not in df.columns:
        raise ValueError("La colonne 'P' (puissance PV) est absente. "
                         "Télécharge le fichier avec pvcalculation=1 (voir download_pvgis.py).")
    df = df.rename(columns={"P": "production"})
    # PVGIS : on ne garde que la production (jeu 2 = série univariée)
    return df[["production"]], "production"


def _assemble_datetime(df: pd.DataFrame) -> pd.Series | None:
    low = {c.lower(): c for c in df.columns}
    need = ["year", "month", "day"]
    if all(n in low for n in need):
        parts = {n: df[low[n]] for n in need}
        if "hour" in low:
            parts["hour"] = df[low["hour"]]
        return pd.to_datetime(pd.DataFrame(parts), errors="coerce")
    return None


def load_generic(path: Path, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Chargeur automatique pour les fichiers Zenodo (structure inconnue a priori)."""
    df = _read_any(path)
    df.columns = [str(c).strip() for c in df.columns]
    tcol = cfg.get("time_col")
    times = None
    if tcol:
        times = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    else:
        for c in df.columns:
            if re.search(r"(date|time|stamp|^ds$)", c.lower()):
                t = pd.to_datetime(df[c], errors="coerce")
                if t.notna().mean() > 0.9:
                    times, tcol = t, c
                    break
        if times is None:
            times = _assemble_datetime(df)
            tcol = "year/month/day/hour"
        if times is None:
            t = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            if t.notna().mean() > 0.9:
                times, tcol = t, df.columns[0]
    if times is None or times.notna().mean() < 0.9:
        raise ValueError(
            "Colonne de date introuvable. Colonnes du fichier : "
            f"{list(df.columns)}\n-> renseigne 'time_col' dans config.py.")
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    num = df.drop(columns=[c for c in [tcol] if c in df.columns]) \
            .apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().mean() > 0.5]
    num.index = times.values
    num = num[~num.index.isna()]
    drop = [c for c in cfg.get("drop_cols", []) if c in num.columns]
    num = num.drop(columns=drop)
    target = cfg.get("target_col")
    if not target:
        # on cherche d'abord un nom qui parle de production / puissance / énergie,
        # puis seulement un nom qui parle de PV / solaire / éolien
        groups = [r"(prod|generation|output|power|puissance|energy|energie|kwh|kw|mw)",
                  r"(pv|solar|photov|wind|eol)"]
        cands = []
        for pat in groups:
            cands = [c for c in num.columns if re.search(pat, c.lower())]
            if cands:
                break
        if len(cands) == 1:
            target = cands[0]
        elif len(cands) > 1:
            target = max(cands, key=lambda c: num[c].std())
            print(f"  [info] plusieurs colonnes candidates {cands}. Cible choisie : '{target}'. "
                  f"Renseigne 'target_col' dans config.py si ce n'est pas la bonne.")
        else:
            raise ValueError(
                f"Colonne cible introuvable. Colonnes numériques : {list(num.columns)}\n"
                "-> renseigne 'target_col' dans config.py.")
    if target not in num.columns:
        raise ValueError(f"target_col='{target}' absente. Colonnes : {list(num.columns)}")
    num = num.rename(columns={target: "production"})
    return num, "production"


def load_pv_orientation(path: Path, cfg: dict) -> tuple[pd.DataFrame, str]:
    """
    Zenodo 19245713 : "PV Generation Dataset with Orientation Diversity" (PV_Data.xlsx).
    Le fichier n'a PAS de ligne d'en-tête et PAS de colonne date : seulement Jour (1-365)
    et Heure (1-24). 21 colonnes, dans l'ordre de la description Zenodo :
      0 jour, 1 cos(jour), 2 heure, 3 cos(heure), 4 température, 5 point de rosée,
      6 vent, 7 pression, 8 condition météo, 9 rayonnement solaire,
      10-19 : pour PV1..PV5 le cosinus de l'angle d'incidence et la puissance mesurée,
      20 : puissance PV totale (= somme des 5 panneaux)  <- CIBLE
    Les 5 puissances des panneaux sont retirées des variables explicatives, car leur somme
    est exactement la cible (ce serait de la triche).
    Si le fichier a une autre forme, on retombe sur le chargeur automatique.
    """
    raw = _read_any_noheader(path)
    if raw.shape[1] != 21 or not np.issubdtype(raw.dtypes.iloc[0], np.number) \
            or raw.iloc[:, :].isna().all(axis=None):
        print("  [info] structure inattendue pour ce fichier : chargeur automatique.")
        return load_generic(path, cfg)
    names = ["jour", "cos_jour", "heure", "cos_heure", "temperature", "point_rosee", "vent",
             "pression", "condition_meteo", "rayonnement"]
    pan = [f"{k}{i}" for i in range(1, 6) for k in ("cos_incidence_pv", "puissance_pv")]
    df = raw.copy()
    df.columns = names + pan + ["production"]
    # identifie les colonnes de puissance : celles dont la somme = production
    tot = df["production"].values
    layouts = {"alterne": [f"puissance_pv{i}" for i in range(1, 6)]}
    ok = [c for c in layouts["alterne"] if c in df.columns]
    if not np.allclose(df[ok].sum(axis=1).values, tot, rtol=1e-3, atol=1e-6):
        # l'ordre est peut-être "5 cosinus puis 5 puissances"
        alt = pan[:]
        df.columns = names + [f"cos_incidence_pv{i}" for i in range(1, 6)] + \
            [f"puissance_pv{i}" for i in range(1, 6)] + ["production"]
        ok = [f"puissance_pv{i}" for i in range(1, 6)]
        if not np.allclose(df[ok].sum(axis=1).values, tot, rtol=1e-3, atol=1e-6):
            print("  [attention] la somme des 5 puissances ne donne pas la colonne 20 : "
                  "vérifie l'ordre des colonnes de ton fichier.")
    # date : année arbitraire (non bissextile), la donnée n'en contient pas
    jour, heure = df["jour"].astype(int), df["heure"].astype(int)
    annee_idx = (jour.diff().fillna(0) < 0).cumsum()
    ts = pd.Timestamp("2023-01-01") + pd.to_timedelta(jour - 1 + 365 * annee_idx, unit="D") \
        + pd.to_timedelta(heure - 1, unit="h")
    df.index = pd.DatetimeIndex(ts)
    drop = ok + ["jour", "heure", "cos_jour", "cos_heure"] + list(cfg.get("drop_cols", []))
    df = df.drop(columns=[c for c in drop if c in df.columns])
    print("  [info] PV_Data : cible = puissance PV totale ; colonnes retirées : " + ", ".join(drop))
    return df, "production"


def _read_any_noheader(path: Path) -> pd.DataFrame:
    if Path(path).suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, header=None)
    return pd.read_csv(path, header=None, sep=None, engine="python")


LOADERS = {"electric": load_electric, "pvgis": load_pvgis, "generic": load_generic,
           "pv_orientation": load_pv_orientation}


# ----------------------------------------------------------------------------
# Construction de l'objet Dataset
# ----------------------------------------------------------------------------
def build_dataset(key: str, df: pd.DataFrame, cfg: dict, target_name: str = "production") -> Dataset:
    df = df.sort_index()
    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        df = df.groupby(level=0).mean()
    freq = infer_freq(df.index)
    if freq == "MS":
        df.index = df.index.to_period("M").to_timestamp()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    n_missing_ts = len(full_idx) - len(df)
    raw_obs = df.reindex(full_idx)

    max_gap = cfg.get("max_gap", 3)
    n_nan_before = int(raw_obs[target_name].isna().sum())
    # interpolation des petits trous seulement (les longs trous restent invalides)
    t_short = _short_gap_interp(raw_obs[target_name], max_gap)
    valid = t_short.notna().values
    n_interp = int(valid.sum() - raw_obs[target_name].notna().sum())
    n_invalid = int((~valid).sum())

    # série "pleine" pour servir d'entrée aux modèles (les longs trous sont comblés
    # par interpolation, mais ces points ne servent jamais de cible ni d'évaluation)
    filled = raw_obs.interpolate(method="time", limit_area="inside").ffill().bfill()
    if cfg.get("clip_negative", False):
        n_neg = int((filled[target_name] < 0).sum())
        filled[target_name] = filled[target_name].clip(lower=0)
    else:
        n_neg = 0
    y = filled[target_name].astype(float)
    y.name = target_name
    y_obs = y.where(valid)
    exog = filled.drop(columns=[target_name]).astype(float)
    # on enlève les variables constantes
    exog = exog.loc[:, exog.std() > 0] if exog.shape[1] else exog
    if not cfg.get("use_exog", False):
        exog = exog.iloc[:, 0:0]

    n = len(y)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    n_test = n - n_train - n_val

    info = dict(
        n_obs=int(n), n_obs_fichier=int(len(df)), freq=freq, freq_label=FREQ_LABEL[freq],
        start=str(full_idx[0]), end=str(full_idx[-1]),
        variables=list(filled.columns), n_missing_timestamps=int(n_missing_ts),
        n_nan_target_apres_reindex=n_nan_before, n_interpolated=n_interp,
        n_invalid_target=n_invalid, n_duplicates=n_dup, n_negative_clipped=n_neg,
        train_period=(str(full_idx[0]), str(full_idx[n_train - 1])),
        val_period=(str(full_idx[n_train]), str(full_idx[n_train + n_val - 1])),
        test_period=(str(full_idx[n_train + n_val]), str(full_idx[-1])),
        exog_cols=list(exog.columns),
    )
    return Dataset(
        key=key, title=cfg["title"], unit=cfg["unit"], freq=freq,
        period=PERIOD_BY_FREQ[freq], horizon=int(cfg["horizon"]),
        target_name=target_name, y=y, y_obs=y_obs, valid=valid, exog=exog,
        cal=calendar_features(full_idx, freq), cal_raw=calendar_raw(full_idx, freq),
        raw=filled, info=info, n_train=n_train, n_val=n_val, n_test=n_test, cfg=cfg)


def load_dataset(key: str) -> Dataset:
    cfg = dict(config.DATASETS[key])
    loader = LOADERS[cfg["loader"]]
    path = find_file(cfg["folder"], cfg.get("file"))
    print(f"  Lecture : {path}")
    df, tname = loader(path, cfg)
    return build_dataset(key, df, cfg, tname)


def inspect_file(key: str) -> None:
    """Affiche les colonnes d'un fichier pour aider à renseigner config.py."""
    cfg = config.DATASETS[key]
    try:
        path = find_file(cfg["folder"], cfg.get("file"))
    except FileNotFoundError as e:
        print(f"[{key}] {e}")
        return
    print(f"[{key}] {path}")
    if cfg["loader"] in ("electric", "pvgis", "pv_orientation"):
        print("   format connu, rien à renseigner.")
        return
    df = _read_any(path)
    print(f"   {len(df)} lignes, colonnes : {list(df.columns)}")
    print(df.head(3).to_string())
