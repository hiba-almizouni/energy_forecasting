"""
Données SYNTHÉTIQUES, uniquement pour tester que le code tourne de bout en bout
(mode --demo). Elles n'ont aucune valeur scientifique et ne doivent jamais être
utilisées dans le rapport final.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def make_electric(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1985-01-01", "2018-01-01", freq="MS")
    t = np.arange(len(idx))
    m = idx.month.values
    trend = 62 + 0.17 * t - 0.00012 * t ** 2
    seas = 9 * np.cos(2 * np.pi * (m - 1) / 12) + 7 * np.cos(4 * np.pi * (m - 7) / 12)
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 2.2)
    y = trend + seas + noise
    pd.DataFrame({"DATE": idx.strftime("%m-%d-%Y"), "IPG2211A2N": y.round(4)}).to_csv(path, index=False)


def make_pv_weather(path: Path, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-15 00:00", periods=7300, freq="h")
    n = len(idx)
    doy = idx.dayofyear.values
    hour = idx.hour.values
    lat = np.deg2rad(36.8)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (doy - 81) / 365)
    omega = np.deg2rad(15 * (hour - 12))
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(omega)
    clear = 1000 * np.clip(cosz, 0, None) ** 1.15
    # nébulosité : composante journalière + horaire
    daily = np.repeat(np.clip(rng.beta(1.2, 3.0, n // 24 + 1) * 1.3, 0, 1), 24)[:n]
    hourly = np.zeros(n)
    for i in range(1, n):
        hourly[i] = 0.8 * hourly[i - 1] + rng.normal(0, 0.12)
    cloud = np.clip(daily + hourly, 0, 1)
    ghi = clear * (1 - 0.75 * cloud)
    temp = 17 + 9 * np.sin(2 * np.pi * (doy - 110) / 365) + 6 * np.sin(2 * np.pi * (hour - 9) / 24) \
        - 3 * cloud + rng.normal(0, 1.2, n)
    hum = np.clip(70 - 25 * np.sin(2 * np.pi * (hour - 9) / 24) + 20 * cloud + rng.normal(0, 5, n), 15, 100)
    wind = np.clip(3 + 2 * cloud + rng.normal(0, 1.2, n), 0, None)
    pv = 5 * (ghi / 1000) * (1 - 0.004 * (temp + 0.02 * ghi - 25)) + rng.normal(0, 0.08, n) * (ghi > 20)
    pv = np.clip(pv, 0, None)
    df = pd.DataFrame({"timestamp": idx, "pv_power_kw": pv.round(4), "ghi_wm2": ghi.round(2),
                       "temp_c": temp.round(2), "humidity_pct": hum.round(1),
                       "wind_ms": wind.round(2), "cloud_cover": cloud.round(3)})
    # trous : quelques petits (1-3 h) et deux longs (2 et 4 jours) pour tester la robustesse
    drop = np.zeros(n, bool)
    for s in rng.choice(np.arange(200, n - 10), 25, replace=False):
        drop[s:s + rng.integers(1, 4)] = True
    drop[3000:3048] = True
    drop[5600:5696] = True
    df = df.loc[~drop]
    df.to_csv(path, index=False)


def make_demo_folder(base: Path) -> Path:
    for sub in ("electric", "pv_weather"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    make_electric(base / "electric" / "Electric_Production.csv")
    make_pv_weather(base / "pv_weather" / "pv_weather.csv")
    return base
