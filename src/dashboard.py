"""
Page web de comparaison (HTML autonome, thème blanc et bleu "météo").
Lit outputs/<jeu>/resultats.json et les figures PNG. Aucun accès internet requis.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import config
from .report import (MODELS, FAMILLE, fr, fp, pct, ranking, ok_models, variants_of, main_variant,
                     vname_fr, txt_results, txt_errors, txt_eda, txt_data, load_results)

COL = {"ARIMA": "#2b6cb0", "Prophet": "#2f9e8f", "XGBoost": "#e08a1e", "LSTM": "#7c5cc4",
       "Naïf saisonnier": "#8a9bb0"}

CSS = """
:root{--bleu:#1d6fd8;--bleu2:#0b4fa8;--ciel:#e8f2fd;--ciel2:#f4f9ff;--txt:#17324d;--gris:#6b7f94;--ok:#1a9e5c;--bord:#d3e3f5}
*{box-sizing:border-box}body{margin:0;font-family:"Segoe UI",Calibri,Arial,sans-serif;color:var(--txt);background:var(--ciel2);line-height:1.5}
header{background:linear-gradient(135deg,#0b4fa8 0%,#1d6fd8 55%,#5aa9f0 100%);color:#fff;padding:34px 28px 26px}
header h1{margin:0 0 6px;font-size:28px}header p{margin:2px 0;opacity:.92}
.badge{display:inline-block;background:rgba(255,255,255,.2);border-radius:20px;padding:3px 12px;font-size:13px;margin:8px 6px 0 0}
.warn{background:#fff3f3;color:#b42318;border:1px solid #f4c7c3;padding:10px 16px;margin:14px 28px;border-radius:10px;font-weight:600}
nav{display:flex;gap:8px;padding:14px 28px 0;flex-wrap:wrap;position:sticky;top:0;background:var(--ciel2);z-index:5}
nav button{border:1px solid var(--bord);background:#fff;color:var(--bleu2);padding:9px 18px;border-radius:22px;cursor:pointer;font-size:15px;font-weight:600}
nav button.on{background:var(--bleu);color:#fff;border-color:var(--bleu)}
main{padding:10px 28px 50px;max-width:1280px;margin:0 auto}
section.tab{display:none}section.tab.on{display:block}
h2{color:var(--bleu2);margin:26px 0 10px;font-size:22px;border-left:5px solid var(--bleu);padding-left:10px}
h3{color:var(--bleu2);margin:18px 0 8px;font-size:17px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.card{background:#fff;border:1px solid var(--bord);border-radius:14px;padding:14px 16px;box-shadow:0 2px 8px rgba(29,111,216,.07)}
.card .k{font-size:12.5px;color:var(--gris);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:22px;font-weight:700;color:var(--bleu2);margin-top:2px}
.card .s{font-size:12.5px;color:var(--gris)}
.box{background:#fff;border:1px solid var(--bord);border-radius:14px;padding:16px 18px;margin:12px 0;box-shadow:0 2px 8px rgba(29,111,216,.06)}
.bars .row{display:flex;align-items:center;gap:10px;margin:7px 0}
.bars .nm{width:130px;font-weight:600;font-size:14px}.bars .tr{flex:1;background:var(--ciel);border-radius:8px;height:22px;overflow:hidden}
.bars .fl{height:100%;border-radius:8px}.bars .vl{width:150px;font-size:13px;color:var(--gris)}
table{border-collapse:collapse;width:100%;font-size:14px;background:#fff}
th{background:var(--ciel);color:var(--bleu2);padding:8px 10px;text-align:center;border:1px solid var(--bord)}
td{padding:7px 10px;border:1px solid var(--bord);text-align:center}td:first-child{text-align:left;font-weight:600}
tr.best td{background:#e6f4ea;font-weight:700}td.sig{background:#e6f4ea;color:#126b3d;font-weight:600}td.ns{background:#fff5e5;color:#8a5a00}
.wrap{overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
figure{margin:0;background:#fff;border:1px solid var(--bord);border-radius:14px;padding:10px}
figure img{width:100%;height:auto;border-radius:8px;cursor:zoom-in}figcaption{font-size:12.5px;color:var(--gris);text-align:center;margin-top:6px}
ul.txt{padding-left:20px}ul.txt li{margin:6px 0}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:700;color:#fff}
#zoom{display:none;position:fixed;inset:0;background:rgba(10,30,60,.85);z-index:20;align-items:center;justify-content:center}
#zoom img{max-width:94vw;max-height:92vh;background:#fff;border-radius:10px}
footer{text-align:center;color:var(--gris);font-size:12.5px;padding:20px}
"""

JS = """
function show(id){document.querySelectorAll('section.tab').forEach(s=>s.classList.toggle('on',s.id===id));
document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));window.scrollTo(0,0)}
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{const z=document.getElementById('zoom');z.querySelector('img').src=i.src;z.style.display='flex'});
document.getElementById('zoom').onclick=e=>e.currentTarget.style.display='none';
show('synthese');
"""


def esc(x):
    return html.escape(str(x))


def img64(path):
    p = Path(path)
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def fig_html(path, cap):
    src = img64(path)
    return f'<figure><img src="{src}" alt="{esc(cap)}"><figcaption>{esc(cap)}</figcaption></figure>' if src else ""


def card(k, v, s=""):
    return f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'


def table(header, rows, best=None, wrap=True):
    h = "".join(f"<th>{esc(x)}</th>" for x in header)
    body = ""
    for i, r in enumerate(rows):
        cls = ' class="best"' if best is not None and i == best else ""
        body += f"<tr{cls}>" + "".join(f"<td{c[1] if isinstance(c, tuple) else ''}>{esc(c[0] if isinstance(c, tuple) else c)}</td>" for c in r) + "</tr>"
    t = f"<table><tr>{h}</tr>{body}</table>"
    return f'<div class="wrap">{t}</div>' if wrap else t


def bars(vr, unit):
    names = ok_models(vr) + ["Naïf saisonnier"]
    vals = {m: vr["models"][m]["test"]["RMSE"] for m in ok_models(vr)}
    vals["Naïf saisonnier"] = vr["naif"]["test"]["RMSE"]
    mx = max(vals.values()) or 1
    order = sorted(names, key=lambda m: vals[m])
    out = '<div class="bars">'
    for m in order:
        w = max(3, 100 * vals[m] / mx)
        out += (f'<div class="row"><div class="nm">{esc(m)}</div><div class="tr"><div class="fl" '
                f'style="width:{w:.1f}%;background:{COL[m]}"></div></div><div class="vl">RMSE {fr(vals[m])} {esc(unit)}</div></div>')
    return out + "</div>"


def variant_block(D, v, fdir):
    vr = D["variants"][v]
    names = ok_models(vr)
    u = D["unit"]
    out = f"<h3>Cas : {esc(vname_fr(v))}</h3>"
    out += f'<div class="box"><b>Classement par RMSE sur le test</b> (plus petit = meilleur){bars(vr, u)}</div>'
    # test
    rows = []
    for m in names:
        t = vr["models"][m]["test"]
        rows.append([m, FAMILLE[m], fr(t["MAE"]), fr(t["RMSE"]), fr(t["R2"]), fr(t["R2_adj"]), fr(t["nRMSE"]),
                     fp(t["p_pente"]), fp(t["p_MZ"]), pct(vr["models"][m]["pi"]["coverage"], 0)])
    nt = vr["naif"]["test"]
    rows.append(["Naïf saisonnier", "référence", fr(nt["MAE"]), fr(nt["RMSE"]), fr(nt["R2"]), fr(nt["R2_adj"]),
                 fr(nt["nRMSE"]), fp(nt["p_pente"]), fp(nt["p_MZ"]), "-"])
    out += "<h3>Performances sur le test</h3>" + table(
        ["Modèle", "Famille", "MAE", "RMSE", "R²", "R² ajusté", "nRMSE", "p (pente)", "p (MZ)", "Couverture 95 %"],
        rows, best=names.index(vr["best_val"]))
    out += '<p style="color:var(--gris);font-size:13px">Ligne verte = modèle retenu sur la validation (le test ne sert pas à choisir).</p>'
    # validation + hyperparamètres
    rows = [[m, vr["models"][m]["label"], fr(vr["models"][m]["val"]["RMSE"]), fr(vr["models"][m]["val"]["MAE"]),
             fr(vr["models"][m]["val"]["R2"])] for m in names]
    out += "<h3>Validation et hyperparamètres retenus</h3>" + table(
        ["Modèle", "Hyperparamètres", "RMSE valid.", "MAE valid.", "R² valid."], rows, best=names.index(vr["best_val"]))
    # DM
    allm = names + ["Naïf saisonnier"]
    rows = []
    for a in allm:
        r = [a]
        for b in allm:
            if a == b:
                r.append("-")
            else:
                p = vr["dm"][a][b]["p"]
                r.append((fp(p), ' class="sig"' if (p is not None and p < 0.05) else ' class="ns"'))
        rows.append(r)
    out += ("<h3>Différences de précision entre modèles (Diebold-Mariano)</h3>"
            "<p style='font-size:13px;color:var(--gris)'>Vert : différence significative (p &lt; 0,05). Orange : non significative.</p>"
            + table(["Modèle 1 \\ 2"] + allm, rows))
    out += '<div class="box"><b>Lecture des résultats</b><ul class="txt">' + "".join(
        f"<li>{esc(t)}</li>" for t in txt_results(D, v)) + "</ul></div>"
    out += '<div class="box"><b>Analyse des erreurs</b><ul class="txt">' + "".join(
        f"<li>{esc(t)}</li>" for t in txt_errors(D, v)) + "</ul></div>"
    figs = [fig_html(p, c) for p, c in vr["figures"]]
    out += '<h3>Figures</h3><div class="grid">' + "".join(figs) + "</div>"
    return out


def dataset_tab(D, outdir):
    i, u = D["info"], D["unit"]
    v0 = main_variant(D)
    vr0 = D["variants"][v0]
    bv = vr0["best_val"]
    t = vr0["models"][bv]["test"]
    out = f'<section class="tab" id="{D["key"]}"><h2>{esc(D["title"])}</h2>'
    out += '<div class="cards">' + "".join([
        card("Observations", i["n_obs"], i["freq_label"]),
        card("Période", i["start"][:10], "au " + i["end"][:10]),
        card("Horizon de prévision", f"{D['horizon']} pas", "origine glissante"),
        card("Modèle retenu", bv, vname_fr(v0)),
        card("RMSE test", f"{fr(t['RMSE'])}", u),
        card("R² test", fr(t["R2"]), "R² ajusté " + fr(t["R2_adj"])),
    ]) + "</div>"
    out += f'<div class="box">{esc(txt_data(D))}</div>'
    out += "<h2>Analyse exploratoire</h2>"
    out += '<div class="box"><ul class="txt">' + "".join(f"<li>{esc(t_)}</li>" for t_ in txt_eda(D)) + "</ul></div>"
    out += '<div class="grid">' + "".join(fig_html(p, c) for p, c in D["eda"]["figures"]) + "</div>"
    out += "<h2>Modèles et résultats</h2>"
    for v in variants_of(D):
        out += variant_block(D, v, outdir)
    if "avec_exog" in D["variants"] and "sans_exog" in D["variants"]:
        a, b = D["variants"]["sans_exog"], D["variants"]["avec_exog"]
        rows = []
        for m in MODELS:
            if m in ok_models(a) and m in ok_models(b):
                r0, r1 = a["models"][m]["test"]["RMSE"], b["models"][m]["test"]["RMSE"]
                g = 1 - r1 / r0
                rows.append([m, fr(r0), fr(r1), (pct(g), ' class="sig"' if g > 0 else ' class="ns"')])
        out += ("<h2>Apport des variables explicatives</h2>"
                + table(["Modèle", "RMSE sans variables", "RMSE avec variables", "Gain relatif"], rows))
    v = main_variant(D)
    vr = D["variants"][v]
    rows = [[m, fr(vr["models"][m]["temps"]["choix_s"], 1), fr(vr["models"][m]["temps"]["ajustement_s"], 1),
             fr(vr["models"][m]["temps"]["prediction_s"], 2)] for m in ok_models(vr)]
    out += "<h2>Coûts de calcul</h2>" + table(
        ["Modèle", "Choix des hyperparamètres (s)", "Ajustement final (s)", "Prédiction du test (s)"], rows)
    return out + "</section>"


def synthese_tab(R):
    out = '<section class="tab" id="synthese"><h2>Synthèse : meilleur modèle par jeu de données</h2><div class="cards">'
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        bv = vr["best_val"]
        t = vr["models"][bv]["test"]
        out += card(D["title"], bv, f"RMSE test {fr(t['RMSE'])} {D['unit']} · R² {fr(t['R2'])}")
    out += "</div>"
    rows = []
    for D in R:
        for v in variants_of(D):
            vr = D["variants"][v]
            bv, bt = vr["best_val"], vr["best_test"]
            rows.append([D["title"], vname_fr(v), bv, fr(vr["models"][bv]["test"]["RMSE"]), bt,
                         fr(vr["models"][bt]["test"]["RMSE"]), fr(vr["naif"]["test"]["RMSE"]),
                         pct(vr["models"][bv]["test"]["nRMSE"])])
    out += "<h2>Tous les cas</h2>" + table(
        ["Jeu", "Cas", "Retenu (validation)", "RMSE test", "Meilleur sur test", "RMSE test", "RMSE naïf", "nRMSE retenu"], rows)
    out += "<h2>Comparaison statistique, Machine Learning et Deep Learning</h2>"
    rows = []
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        r = [D["title"]]
        for m in MODELS:
            r.append(fr(vr["models"][m]["test"]["RMSE"]) if m in ok_models(vr) else "n/a")
        r.append(ranking(vr)[0])
        rows.append(r)
    out += table(["Jeu", "ARIMA (stat.)", "Prophet (stat.)", "XGBoost (ML)", "LSTM (DL)", "Meilleur sur test"], rows)
    out += ('<div class="box"><b>À retenir</b><ul class="txt">'
            "<li>Le modèle est choisi sur la validation. Le test sert seulement à l'évaluation finale.</li>"
            "<li>Le modèle naïf saisonnier est la référence : un modèle qui ne le bat pas n'apporte rien.</li>"
            "<li>Les p-values viennent de la régression réel = a + b × prévu (écarts-types robustes). "
            "p (MZ) teste l'absence de biais.</li>"
            "<li>Les variables météo sont supposées connues à l'avance : c'est une hypothèse favorable.</li></ul></div>")
    return out + "</section>"


def build_dashboard(outdir: Path, keys, demo=False, path: Path | None = None) -> Path:
    outdir = Path(outdir)
    R = load_results(outdir, keys)
    fast = any(D.get("fast_mode") for D in R)
    h = ['<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>Energy Forecasting - Comparaison des résultats</title>", f"<style>{CSS}</style></head><body>",
         "<header><h1>Energy Forecasting</h1>"
         "<p>Comparaison d'ARIMA, Prophet, XGBoost et LSTM sur deux séries de production d'énergie</p>"
         f"<span class='badge'>{esc(', '.join(config.AUTHORS))}</span><span class='badge'>{esc(config.SCHOOL)}</span>"
         "<span class='badge'>Découpage chronologique 70 / 15 / 15</span></header>"]
    if demo:
        h.append('<div class="warn">VERSION DE TEST : données synthétiques. Ces résultats n\'ont aucune valeur. Ne pas rendre.</div>')
    if fast:
        h.append('<div class="warn">Exécution en mode rapide : grilles d\'hyperparamètres réduites.</div>')
    h.append("<nav><button data-t='synthese' onclick=\"show('synthese')\">Synthèse</button>" + "".join(
        f"<button data-t='{D['key']}' onclick=\"show('{D['key']}')\">{esc(D['title'])}</button>" for D in R) + "</nav><main>")
    h.append(synthese_tab(R))
    for D in R:
        h.append(dataset_tab(D, outdir))
    h.append('</main><footer>Page générée automatiquement à partir des résultats du projet. '
             'Clique sur une figure pour l\'agrandir.</footer><div id="zoom"><img alt=""></div>'
             f"<script>{JS}</script></body></html>")
    path = Path(path) if path else outdir / ("Comparaison_Resultats_DEMO.html" if demo else "Comparaison_Resultats.html")
    path.write_text("".join(h), encoding="utf-8")
    return path
