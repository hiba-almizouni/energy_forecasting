"""
Configuration du projet Energy Forecasting.
Tout ce que l'on peut vouloir changer se trouve ici.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# ----------------------------------------------------------------------------
# Informations pour la page de garde du rapport
# ----------------------------------------------------------------------------
AUTHORS = ["Babi Mariam", "Almizouni Hiba"]
UNIVERSITY = "Université de Sousse"
SCHOOL = "Ecole Nationale d'Ingénieurs de Sousse"
FILIERE = "Génie Informatique Appliquée"
ENCADRANT = "Tarek Aroui"
ANNEE_UNIV = "2025-2026"
REF = "…../…………"
COURSE = "Prévision des séries temporelles - Energy Forecasting"

# ----------------------------------------------------------------------------
# Choix des deux jeux de données (parmi : electric, pvgis, pv_weather, wind)
# ----------------------------------------------------------------------------
SELECTED_DATASETS = ["electric", "pv_weather"]

# ----------------------------------------------------------------------------
# Paramètres généraux
# ----------------------------------------------------------------------------
SEED = 42
FAST_MODE = False        # True = grilles et epochs réduits (test rapide du code)
TRAIN_RATIO = 0.70       # imposé par le sujet
VAL_RATIO = 0.15         # imposé par le sujet (le reste = test, 15 %)
ALPHA_PI = 0.05          # intervalles de prédiction à 95 %

# ----------------------------------------------------------------------------
# Description des jeux de données
#   horizon      : nombre de pas prédits à chaque origine (rolling origin)
#                  mensuel -> 12 mois ; horaire -> 24 h (prévision "jour J+1")
#   folder/file  : où trouver le fichier dans data/
#   time_col / target_col : laisser None pour la détection automatique
#   drop_cols    : colonnes à retirer (ex. identifiants, colonnes qui
#                  contiennent déjà la production = fuite d'information)
#   use_exog     : True = on compare aussi le modèle avec variables explicatives
#   max_gap      : trous de moins de max_gap pas = interpolés ; plus longs = non utilisés
# ----------------------------------------------------------------------------
DATASETS = {
    "electric": dict(
        title="Production électrique (Kaggle)",
        folder="electric", file="Electric_Production.csv",
        loader="electric", horizon=12, unit="indice de production",
        time_col=None, target_col=None, drop_cols=[], use_exog=False,
        max_gap=3, clip_negative=False,
        prophet_weekly=False,
    ),
    "pvgis": dict(
        title="Production photovoltaïque (PVGIS)",
        folder="pvgis", file=None,          # None = premier .csv du dossier
        loader="pvgis", horizon=24, unit="W",
        time_col=None, target_col=None, drop_cols=[], use_exog=False,
        max_gap=6, clip_negative=True,
        prophet_weekly=False,
    ),
    "pv_weather": dict(
        title="Production photovoltaïque et météo (Zenodo)",
        folder="pv_weather", file=None,
        loader="pv_orientation", horizon=24, unit="W",   # chargeur dédié à PV_Data.xlsx
        time_col=None, target_col=None,      # <- à renseigner si la détection échoue
        drop_cols=[], use_exog=True,
        max_gap=6, clip_negative=True,
        prophet_weekly=False,
    ),
    "wind": dict(
        title="Production éolienne (Zenodo)",
        folder="wind", file=None,
        loader="generic", horizon=24, unit="MW",
        time_col=None, target_col=None, drop_cols=[], use_exog=False,
        max_gap=6, clip_negative=True,
        prophet_weekly=True,
    ),
}

# ----------------------------------------------------------------------------
# Téléchargement PVGIS (jeu de données 2) - Tunis par défaut
# ----------------------------------------------------------------------------
PVGIS = dict(lat=36.8065, lon=10.1815, startyear=2016, endyear=2020,
             peakpower=1.0, loss=14, angle=30, aspect=0)

# ----------------------------------------------------------------------------
# Hyperparamètres : grilles explorées sur le jeu de VALIDATION uniquement
# ----------------------------------------------------------------------------
LSTM_SEQ_LEN = {"MS": 24, "h": 72, "D": 28, "30min": 96, "15min": 96}
LSTM_MAX_EPOCHS = 150
LSTM_PATIENCE = 15
XGB_MAX_ROUNDS = 2000
XGB_EARLY_STOP = 50
