# Projet Energy Forecasting : ARIMA, Prophet, XGBoost, LSTM

Ce projet traite le sujet `Projet_forecasting.docx` : EDA, quatre modèles, évaluation (MAE, RMSE, R²,
R² ajusté, p-values), analyse comparative, et génération automatique du rapport Word.

Jeux choisis par défaut : **1 (Production électrique, mensuel)** et **3 (PV + météo, horaire)**.
Pour en changer : `SELECTED_DATASETS` dans `config.py` (`electric`, `pvgis`, `pv_weather`, `wind`).

## Étapes d'exécution

1. **Python 3.10 à 3.12** recommandé. Ouvrir un terminal dans ce dossier.
2. Environnement et dépendances :
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/Mac)
   pip install -r requirements.txt
   ```
3. Télécharger les données et les placer dans `data/` (voir `data/README.txt`) :
   - Kaggle : https://www.kaggle.com/datasets/shenba/time-series-datasets (fichier `Electric_Production.csv`)
   - Zenodo PV + météo : https://zenodo.org/records/19245713
   - PVGIS : `python download_pvgis.py`, ou https://re.jrc.ec.europa.eu/pvg_tools/en/
   - Zenodo éolien : https://zenodo.org/records/8340501
4. **Vérifier les colonnes** des fichiers Zenodo :
   ```
   python run_all.py --inspect
   ```
   Le code détecte seul la date et la cible. Si le message demande de préciser, remplir `time_col`,
   `target_col` (et `drop_cols` pour retirer une colonne qui contient déjà la production) dans
   `config.py`, section du jeu concerné.
5. Essai rapide (quelques minutes) pour vérifier que tout marche :
   ```
   python run_all.py --fast
   ```
6. Exécution complète (de 20 min à 2 h selon la machine ; le LSTM est le plus long) :
   ```
   python run_all.py
   ```
7. Résultats dans `outputs/` : `<jeu>/figures/*.png`, `<jeu>/predictions_test_*.csv`,
   `<jeu>/metriques_test_*.csv`, `<jeu>/resultats.json` et **`Rapport_Energy_Forecasting.docx`**.
8. Relire le rapport, ajouter ton analyse personnelle et corriger si besoin. Le texte est généré à
   partir des chiffres. Pour le régénérer sans tout recalculer : `python run_all.py --report-only`.

`python run_all.py --demo` teste le code sur des données **synthétiques** (rapport marqué TEST). À ne jamais rendre.

## Méthode en bref

- Découpage chronologique 70/15/15. Le test n'intervient ni dans l'entraînement, ni dans les hyperparamètres, ni dans le choix du modèle.
- Chaque modèle : grille d'hyperparamètres notée sur la validation, puis ré-ajustement sur train + validation, puis prédiction unique du test.
- Prévision par origine glissante : horizon 12 mois (mensuel) ou 24 h (horaire), réglable (`horizon` dans `config.py`).
- Jeux avec variables explicatives : chaque modèle est lancé avec et sans (étude de l'apport de la météo).
- Référence : modèle naïf saisonnier.
- Statistiques : p-values de la régression réel/prévu (HAC), test de Mincer-Zarnowitz, Ljung-Box, Diebold-Mariano, couverture des intervalles à 95 %.

## Structure

```
config.py            choix des jeux, horizons, auteurs, paramètres
run_all.py           point d'entrée
download_pvgis.py    téléchargement PVGIS
src/data_loader.py   lecture, nettoyage, trous, calendrier
src/eda.py           analyse exploratoire
src/models.py        ARIMA, Prophet, XGBoost, LSTM, naïf
src/metrics.py       MAE, RMSE, R², R² ajusté, p-values, DM
src/pipeline.py      orchestration, figures, résultats
src/report.py        rapport Word automatique
```

## Points d'attention

- La structure exacte des fichiers Zenodo n'a pas pu être vérifiée au moment de l'écriture du code : lance `--inspect` d'abord.
- Si le jeu PV + météo contient des colonnes qui sont déjà de la production (ex. DC et AC), garde une seule cible et mets les autres dans `drop_cols`.
- Les données horaires couvrent moins de deux ans : pas de saisonnalité annuelle apprenable, et le test tombe sur une autre saison que le train. À discuter dans le rapport.
- La météo est supposée connue sur l'horizon (hypothèse de prévision parfaite).
- Le LSTM dépend de la graine et de la machine : de petits écarts d'un lancement à l'autre sont normaux.
- Problème Prophet/cmdstan sous Windows : `pip install --upgrade prophet cmdstanpy`.
