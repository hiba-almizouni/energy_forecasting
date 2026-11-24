"""
Télécharge des données horaires PVGIS (jeu 2) pour un site donné.
Paramètres dans config.PVGIS (par défaut : Tunis, 2016-2020).
Usage : python download_pvgis.py
Si le téléchargement échoue, utilise le site https://re.jrc.ec.europa.eu/pvg_tools/en/
(onglet "Hourly data", cocher "PV power", exporter en CSV) et mets le fichier dans data/pvgis/.
"""
import sys
from pathlib import Path

import requests

import config

P = config.PVGIS
out = config.DATA_DIR / "pvgis"
out.mkdir(parents=True, exist_ok=True)
params = dict(lat=P["lat"], lon=P["lon"], startyear=P["startyear"], endyear=P["endyear"],
              pvcalculation=1, peakpower=P["peakpower"], loss=P["loss"], angle=P["angle"],
              aspect=P["aspect"], outputformat="csv")
for version in ("v5_3", "v5_2"):
    url = f"https://re.jrc.ec.europa.eu/api/{version}/seriescalc"
    try:
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 200 and r.text.lower().find("time,p") >= 0:
            f = out / "pvgis_hourly.csv"
            f.write_text(r.text, encoding="utf-8")
            print(f"OK ({version}) -> {f}")
            sys.exit(0)
        print(f"{version} : code {r.status_code} - {r.text[:120]!r}")
    except Exception as e:
        print(f"{version} : {e}")
print("Échec. Télécharge le fichier à la main (voir l'en-tête de ce script).")
sys.exit(1)
