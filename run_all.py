"""
Point d'entrée du projet.

Exemples :
    python run_all.py --inspect              # affiche les colonnes des fichiers de data/
    python run_all.py --fast                 # essai rapide (grilles réduites)
    python run_all.py                        # exécution complète + rapport Word
    python run_all.py --report-only          # refait seulement le rapport
    python run_all.py --demo                 # test du code sur données SYNTHÉTIQUES
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Energy Forecasting : ARIMA, Prophet, XGBoost, LSTM")
    ap.add_argument("--datasets", nargs="+", choices=list(config.DATASETS),
                    help="jeux de données à traiter (défaut : config.SELECTED_DATASETS)")
    ap.add_argument("--demo", action="store_true", help="données synthétiques (test du code seulement)")
    ap.add_argument("--fast", action="store_true", help="grilles et epochs réduits")
    ap.add_argument("--inspect", action="store_true", help="afficher les colonnes des fichiers")
    ap.add_argument("--report-only", action="store_true", help="ne refaire que le rapport Word")
    ap.add_argument("--no-report", action="store_true", help="ne pas générer le rapport Word")
    args = ap.parse_args()

    if args.fast:
        config.FAST_MODE = True
    datasets = args.datasets or list(config.SELECTED_DATASETS)
    if args.demo:
        from src import synthetic
        base = config.ROOT / "demo_data"
        synthetic.make_demo_folder(base)
        config.DATA_DIR = base
        config.OUTPUT_DIR = config.ROOT / "outputs_demo"
        datasets = ["electric", "pv_weather"]
        print("MODE DÉMO : données synthétiques. Ne pas utiliser pour le rapport final.")

    from src import data_loader as dl
    if args.inspect:
        for k in config.DATASETS:
            dl.inspect_file(k)
        return

    out = Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not args.report_only:
        from src import pipeline
        for k in datasets:
            pipeline.run_dataset(k, out)
    if not args.no_report:
        from src import report
        path = report.build_report(out, datasets, demo=args.demo)
        print(f"\nRapport Word : {path}")
        try:
            from src import dashboard
            web = dashboard.build_dashboard(out, datasets, demo=args.demo)
            print(f"Page web de comparaison : {web}")
        except Exception as e:  # la page web est un complément : ne bloque pas le rapport
            print(f"[attention] page web non générée : {e}")
    print(f"Durée totale : {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
