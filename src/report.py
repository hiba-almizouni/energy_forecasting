"""
Génère le rapport Word (.docx) à partir des fichiers outputs/<jeu>/resultats.json.
Tous les chiffres du texte viennent des résultats : rien n'est écrit à la main.
Relis le rapport, ajoute tes propres commentaires et adapte le texte avant de le rendre.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import config

MODELS = ["ARIMA", "Prophet", "XGBoost", "LSTM"]
FAMILLE = {"ARIMA": "statistique", "Prophet": "statistique", "XGBoost": "Machine Learning",
           "LSTM": "Deep Learning"}
PLURIEL = {"horaire": "horaires", "mensuelle": "mensuelles", "journalière": "journalières",
           "30 minutes": "toutes les 30 minutes", "15 minutes": "toutes les 15 minutes"}

MODEL_DESC = {
    "ARIMA": ("ARIMA décrit une série par ses valeurs passées (partie AR), ses erreurs passées (partie MA) "
              "et des différenciations (partie I). Pour la série mensuelle, on utilise la version "
              "saisonnière SARIMA (période 12). Pour une série horaire, on ajoute des termes de Fourier "
              "qui représentent le cycle journalier. À chaque origine, le modèle est mis à jour avec les "
              "valeurs observées, sans ré-estimer ses coefficients."),
    "Prophet": ("Prophet sépare la série en une tendance, des saisonnalités et, si on le souhaite, "
                "l'effet de variables explicatives. Il ne regarde que la date. Il n'utilise donc pas les "
                "valeurs récentes pour corriger sa prévision."),
    "XGBoost": ("XGBoost construit des arbres de décision l'un après l'autre. Chaque arbre corrige les "
                "erreurs des précédents. On lui donne des retards (lags) de la production, des moyennes "
                "glissantes, des variables de calendrier et, quand elles existent, les variables météo. "
                "Tous les retards sont d'au moins h pas : le modèle n'utilise jamais une valeur qui ne "
                "serait pas connue à l'origine de la prévision."),
    "LSTM": ("Le LSTM est un réseau de neurones récurrent. Il lit une fenêtre de valeurs passées et prédit "
             "les h pas suivants d'un seul coup. Les variables de calendrier (et la météo) de l'horizon "
             "sont ajoutées avant la couche de sortie. Les données sont standardisées avec la moyenne et "
             "l'écart-type du jeu d'entraînement."),
}
AVANTAGES = {
    "ARIMA": ("Rapide, interprétable, intervalles de prédiction fournis par le modèle, très bon sur les "
              "séries régulières.", "Linéaire. Demande un choix d'ordres. Gère mal les ruptures et les "
              "relations non linéaires. Peu adapté aux cycles multiples."),
    "Prophet": ("Facile à utiliser, gère tendance, saisonnalités et variables explicatives, robuste aux "
                "trous.", "N'utilise pas les valeurs récentes. Peut manquer des variations rapides. "
                "Les saisonnalités annuelles demandent au moins deux ans de données."),
    "XGBoost": ("Capte les relations non linéaires et les interactions entre variables. Rapide à "
                "entraîner. Donne l'importance des variables.", "Ne sait pas extrapoler une tendance "
                "hors de la plage vue à l'entraînement. Demande un travail sur les variables d'entrée. "
                "Pas d'intervalle natif."),
    "LSTM": ("Apprend seul les dépendances temporelles et les relations non linéaires. Prédit tout "
             "l'horizon d'un coup.", "Demande beaucoup de données, un réglage fin, du temps de calcul et "
             "une graine fixe pour être reproductible. Moins facile à interpréter."),
}


# ----------------------------------------------------------------------------
# Formatage
# ----------------------------------------------------------------------------
def fr(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{d}f}".replace(".", ",")


def fp(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "n/a"
    return "< 0,001" if p < 0.001 else fr(p, 3)


def peq(p):
    f = fp(p)
    return "p " + f if f.startswith("<") else "p = " + f


def pct(x, d=1):
    return "n/a" if x is None else fr(100 * x, d) + " %"


def liste(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " et " + items[-1]


# ----------------------------------------------------------------------------
# Outils python-docx
# ----------------------------------------------------------------------------
class Doc:
    def __init__(self):
        self.d = Document()
        self.nfig = 0
        self.ntab = 0
        sec = self.d.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.3)
        sec.top_margin = sec.bottom_margin = Cm(2.3)
        st = self.d.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.12
        for name, size, color in [("Heading 1", 17, "1F3864"), ("Heading 2", 14, "2F5496"),
                                  ("Heading 3", 12, "2F5496")]:
            h = self.d.styles[name]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = RGBColor.from_string(color)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
            h.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
        self._footer()

    def _footer(self):
        p = self.d.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for tag, text in [("begin", None), (None, "PAGE"), ("end", None)]:
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)
        run.font.size = Pt(9)

    def h(self, text, level=1):
        return self.d.add_heading(text, level=level)

    def p(self, text, bold=False, italic=False, size=None, align=None, color=None, after=None):
        para = self.d.add_paragraph()
        r = para.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if after is not None:
            para.paragraph_format.space_after = Pt(after)
        return para

    def bullets(self, items):
        for it in items:
            para = self.d.add_paragraph(style="List Bullet")
            para.paragraph_format.space_after = Pt(2)
            para.add_run(it)

    def caption(self, kind, text):
        if kind == "Figure":
            self.nfig += 1
            n = self.nfig
        else:
            self.ntab += 1
            n = self.ntab
        para = self.d.add_paragraph()
        r = para.add_run(f"{kind} {n} - {text}")
        r.italic = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor.from_string("404040")
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return para

    def figure(self, path, caption, width=15.8):
        if not Path(path).exists():
            return
        self.d.add_picture(str(path), width=Cm(width))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.d.paragraphs[-1].paragraph_format.keep_with_next = True
        self.caption("Figure", caption)

    def table(self, header, rows, caption=None, widths=None, size=9, bold_rows=(), align_first_left=True):
        if caption:
            c = self.caption("Tableau", caption)
            c.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(htxt))
            run.bold = True
            run.font.size = Pt(size)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "D9E2F3")
            tcPr.append(shd)
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v))
                run.font.size = Pt(size)
                run.bold = ri in bold_rows
                cells[i].paragraphs[0].alignment = (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and align_first_left)
                                                    else WD_ALIGN_PARAGRAPH.CENTER)
        for row in t.rows:
            for c in row.cells:
                for para in c.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    para.paragraph_format.space_before = Pt(1)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def page_break(self):
        self.d.add_page_break()

    def save(self, path):
        self.d.save(str(path))


# ----------------------------------------------------------------------------
# Lecture des résultats
# ----------------------------------------------------------------------------
def load_results(outdir: Path, keys):
    res = []
    for k in keys:
        f = Path(outdir) / k / "resultats.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} introuvable. Lance d'abord : python run_all.py")
        res.append(json.load(open(f, encoding="utf-8")))
    return res


def variants_of(D):
    order = ["univarie", "sans_exog", "avec_exog"]
    return [v for v in order if v in D["variants"]]


def ok_models(vr):
    return [m for m in MODELS if m in vr["models"] and "test" in vr["models"][m]]


def ranking(vr, split="test", metric="RMSE"):
    ms = ok_models(vr)
    return sorted(ms, key=lambda m: vr["models"][m][split][metric])


def main_variant(D):
    """Variante principale : avec variables explicatives si elles existent."""
    return "avec_exog" if "avec_exog" in D["variants"] else variants_of(D)[0]


def vname_fr(v):
    return {"univarie": "série seule", "sans_exog": "sans variables explicatives",
            "avec_exog": "avec variables explicatives"}[v]


# ----------------------------------------------------------------------------
# Textes automatiques
# ----------------------------------------------------------------------------
def txt_data(D):
    i = D["info"]
    adj = PLURIEL.get(i["freq_label"], i["freq_label"])
    s = [f"Le jeu « {D['title']} » contient {i['n_obs']} observations {adj}, "
         f"du {i['start'][:10]} au {i['end'][:10]}."]
    if i["n_missing_timestamps"] > 0:
        part = i["n_missing_timestamps"] / i["n_obs"]
        s.append(f"Dans le fichier, {i['n_missing_timestamps']} dates manquent par rapport à une grille "
                 f"régulière ({pct(part)}). Les trous courts (au plus {D.get('max_gap', 'quelques')} pas) "
                 f"sont interpolés : {i['n_interpolated']} valeurs. Les {i['n_invalid_target']} valeurs "
                 f"restantes ne servent ni à l'apprentissage ni à l'évaluation.")
    else:
        s.append("Il n'y a aucune date manquante.")
    if i["n_duplicates"]:
        s.append(f"{i['n_duplicates']} dates en double ont été moyennées.")
    if i["n_negative_clipped"]:
        s.append(f"{i['n_negative_clipped']} valeurs négatives ont été ramenées à zéro (une production ne peut pas être négative).")
    return " ".join(s)


def level(x):
    return "forte" if x >= 0.7 else ("modérée" if x >= 0.3 else "faible")


def txt_eda(D):
    E, u = D["eda"], D["unit"]
    out = []
    st = E["stats"]
    out.append(f"La production moyenne est de {fr(st['moyenne'], 2)} {u} (écart-type {fr(st['ecart_type'], 2)}). "
               f"Elle varie de {fr(st['min'], 2)} à {fr(st['max'], 2)} {u}. "
               f"Le coefficient d'asymétrie vaut {fr(st['asymetrie'], 2)}.")
    if st["part_zeros"] > 0.05:
        out.append(f"{pct(st['part_zeros'], 0)} des valeurs sont nulles. Ce sont les heures de nuit : la distribution est très étalée et fortement concentrée en zéro.")
    stl = E.get("stl", {})
    if "force_tendance" in stl:
        out.append(f"La décomposition STL donne une tendance {level(stl['force_tendance'])} "
                   f"(force {fr(stl['force_tendance'], 2)}) et une saisonnalité {level(stl['force_saison'])} "
                   f"(force {fr(stl['force_saison'], 2)}). L'amplitude saisonnière est de {fr(stl['amplitude_saison'], 2)} {u}.")
        if D["freq"] == "MS":
            out.append(f"La tendance passe de {fr(stl['tendance_debut'], 1)} à {fr(stl['tendance_fin'], 1)} {u}, "
                       f"soit {fr(stl['pente_par_an'], 2)} {u} par an en moyenne.")
        else:
            out.append("La période couverte est courte (moins de deux ans) : la « tendance » STL suit surtout le changement de saison, pas une vraie évolution de long terme.")
    a, k = E["adf"], E["kpss"]
    stationnaire = a["p"] is not None and a["p"] < 0.05 and (k["p"] is None or k["p"] >= 0.05)
    out.append(f"Test ADF : {peq(a['p'])} ; test KPSS : {peq(k['p'])}. "
               + ("Les deux tests vont dans le sens d'une série stationnaire."
                  if stationnaire else
                  "Les tests ne permettent pas de conclure à une série stationnaire sans transformation."
                  ) + f" Après différenciation, l'ADF donne {peq(E['adf_diff1']['p'])}.")
    an = E["anomalies"]
    if an["n"] > 0:
        top = ", ".join(f"{d[:10]}" for d, _, _ in an["top"][:3])
        out.append(f"{an['n']} points atypiques ({pct(an['part'])} des valeurs) sont détectés ({an['seuil']} sur le résidu STL). "
                   f"Les plus marqués : {top}. Nous les avons gardés : ce sont des valeurs observées, pas des erreurs prouvées.")
    else:
        out.append("Aucun point atypique n'est détecté avec le critère utilisé.")
    ct = E.get("corr_target", {})
    if ct:
        top = list(ct.items())[:3]
        out.append("Corrélations les plus fortes avec la production : " +
                   liste(f"{k} ({fr(v, 2)})" for k, v in top) + ".")
    return out


def txt_results(D, v):
    vr = D["variants"][v]
    rk = ranking(vr)
    bt, bv = vr["best_test"], vr["best_val"]
    m = vr["models"]
    out = []
    out.append(f"Sur le jeu de test, le plus faible RMSE est obtenu par {bt} "
               f"({fr(m[bt]['test']['RMSE'])} {D['unit']}). L'ordre complet est : " +
               " < ".join(f"{k} ({fr(m[k]['test']['RMSE'])})" for k in rk) + ".")
    if bv == bt:
        out.append(f"{bv} est aussi le meilleur sur la validation. Le choix fait sans toucher au test est donc confirmé.")
    else:
        gap = m[bv]["test"]["RMSE"] / m[bt]["test"]["RMSE"] - 1
        out.append(f"Le meilleur modèle sur la validation est {bv}, mais {bt} fait mieux sur le test. "
                   f"L'écart de RMSE test entre les deux est de {pct(gap)}. Le modèle retenu officiellement reste {bv}, "
                   f"car le test ne doit pas servir à choisir.")
    nv = vr["naif"]["test"]["RMSE"]
    better = [k for k in rk if m[k]["test"]["RMSE"] < nv]
    worse = [k for k in rk if m[k]["test"]["RMSE"] >= nv]
    s = f"La référence naïve (valeur de la même saison) a un RMSE de {fr(nv)}. "
    if better:
        s += f"Elle est battue par {liste(better)}. "
    if worse:
        s += f"Elle n'est pas battue par {liste(worse)}."
    out.append(s.strip())
    sig = [k for k in rk if (m[k]["test"]["p_pente"] or 1) < 0.05]
    nosig = [k for k in rk if k not in sig]
    s = "Dans la régression réel = a + b × prévu, la pente b est significativement différente de 0 (p < 0,05) pour " + \
        (liste(sig) if sig else "aucun modèle") + "."
    if nosig:
        s += f" Elle ne l'est pas pour {liste(nosig)} : leurs prévisions n'apportent pas d'information significative."
    out.append(s)
    unb = [k for k in rk if (m[k]["test"]["p_MZ"] or 0) >= 0.05]
    out.append("Le test de Mincer-Zarnowitz (a = 0 et b = 1) ne rejette pas l'absence de biais pour " +
               (liste(unb) if unb else "aucun modèle") + ". " +
               ("Pour les autres, la prévision est biaisée ou sur/sous-dispersée." if len(unb) < len(rk) else ""))
    # Diebold-Mariano
    sec = [k for k in rk if k != bv]
    parts = []
    for k in sec:
        d = vr["dm"][bv][k]
        if d["p"] is None:
            continue
        parts.append(f"{bv} contre {k} : {peq(d['p'])} ({'différence significative' if d['p'] < 0.05 else 'différence non significative'})")
    if parts:
        out.append("Test de Diebold-Mariano (erreur quadratique) : " + " ; ".join(parts) + ".")
    return out


def txt_errors(D, v):
    vr = D["variants"][v]
    out = []
    for k in ranking(vr):
        ea, mt = vr["erreurs"][k], vr["models"][k]["test"]
        g = ea["mae_par_groupe"]
        wk = max(g, key=g.get)
        step = ea["mae_par_pas"]
        s1, sl = step[min(step, key=lambda x: int(x))], step[max(step, key=lambda x: int(x))]
        sens = "sous-estime" if (ea["biais"] or 0) > 0 else "surestime"
        out.append(f"{k} : biais moyen de {fr(ea['biais'])} {D['unit']} (le modèle {sens} en moyenne). "
                   f"L'erreur absolue la plus forte par {ea['groupe']} est à {ea['groupe']} {wk} "
                   f"({fr(g[wk])}). La MAE passe de {fr(s1)} au premier pas à {fr(sl)} au dernier pas de l'horizon. "
                   f"Le test de Ljung-Box sur les résidus donne {peq(mt['p_LjungBox'])}"
                   + (" : il reste de l'autocorrélation dans les erreurs." if (mt["p_LjungBox"] or 1) < 0.05
                      else " : pas d'autocorrélation significative."))
    b = vr["best_val"]
    pires = vr["erreurs"][b]["pires"][:3]
    out.append(f"Les trois plus grosses erreurs de {b} ont lieu le " +
               liste(f"{p['date'][:16]} (réel {fr(p['reel'], 2)}, prévu {fr(p['prevu'], 2)})" for p in pires) + ".")
    return out


# ----------------------------------------------------------------------------
# Construction du rapport
# ----------------------------------------------------------------------------
def build_report(outdir: Path, keys, demo=False, path: Path | None = None) -> Path:
    outdir = Path(outdir)
    R = load_results(outdir, keys)
    fast = any(D.get("fast_mode") for D in R)
    doc = Doc()
    d = doc

    # ---------------- Page de garde ----------------
    t = d.d.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    left, right = t.rows[0].cells
    left.width = right.width = Cm(8.2)
    left.text = ""
    right.text = ""
    r1 = left.paragraphs[0].add_run(f"Réf : {config.REF}")
    r1.font.size = Pt(11)
    right.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r2 = right.paragraphs[0].add_run(f"AU : {config.ANNEE_UNIV}")
    r2.font.size = Pt(11)
    r2.bold = True
    d.p("", after=18)
    d.p(config.UNIVERSITY, bold=True, size=18, align="center", color="1F3864", after=4)
    d.p(config.SCHOOL, bold=True, size=15, align="center", color="2F5496")
    d.p("", after=26)
    d.p("Projet de Prévision des séries temporelles", bold=True, size=26, align="center", color="1F3864", after=2)
    d.p("Energy Forecasting", bold=True, size=22, align="center", color="2F5496")
    d.p("Comparaison d'ARIMA, Prophet, XGBoost et LSTM sur deux séries de production d'énergie",
        size=13, align="center", italic=True)
    d.p("Jeux de données : " + " et ".join(D["title"] for D in R), size=11, align="center", color="404040")
    d.p("", after=22)
    d.p("Réalisé par :", bold=True, size=13, color="1F3864", after=2)
    for a_ in config.AUTHORS:
        d.p(a_, size=13, after=0)
    d.p("", after=6)
    d.p("Filière :", bold=True, size=13, color="1F3864", after=2)
    d.p(config.FILIERE, size=13)
    d.p("", after=6)
    d.p("Encadrant :", bold=True, size=13, color="1F3864", after=2)
    d.p(config.ENCADRANT, size=13)
    if demo:
        d.p("", after=8)
        d.p("VERSION DE TEST - DONNÉES SYNTHÉTIQUES - NE PAS RENDRE", bold=True, size=14,
            align="center", color="C00000")
        d.p("Ces résultats viennent de séries inventées qui servent seulement à vérifier que le code "
            "fonctionne. Ils n'ont aucune valeur scientifique.", size=10, align="center", color="C00000")
    if fast:
        d.p("", after=2)
        d.p("Exécuté en mode rapide (grilles d'hyperparamètres réduites).", size=10, align="center", color="C00000")
    d.page_break()

    # ---------------- Plan ----------------
    d.h("Plan du rapport", 1)
    d.bullets(["Résumé", "1. Introduction", "2. Méthodologie",
               *[f"{3 + i}. Jeu de données {i + 1} : {D['title']}" for i, D in enumerate(R)],
               "5. Analyse comparative et discussion", "6. Conclusion",
               "Annexe A. Résultats de la phase de validation", "Annexe B. Reproductibilité"])

    # ---------------- Résumé ----------------
    d.h("Résumé", 1)
    d.p("Ce projet compare quatre méthodes de prévision (ARIMA, Prophet, XGBoost, LSTM) sur deux séries "
        "de production d'énergie. Les données sont découpées dans l'ordre du temps (70 % entraînement, "
        "15 % validation, 15 % test). Les hyperparamètres sont choisis sur la validation. Le test n'est "
        "utilisé qu'à la fin.")
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        bv, bt = vr["best_val"], vr["best_test"]
        m = vr["models"]
        d.p(f"{D['title']} : le modèle retenu sur la validation est {bv} "
            f"(RMSE test = {fr(m[bv]['test']['RMSE'])} {D['unit']}, MAE = {fr(m[bv]['test']['MAE'])}, "
            f"R² = {fr(m[bv]['test']['R2'])}, {vname_fr(v)}). "
            + (f"C'est aussi le meilleur sur le test." if bv == bt else
               f"Sur le test, {bt} obtient le plus faible RMSE ({fr(m[bt]['test']['RMSE'])})."))
    d.page_break()

    # ---------------- 1. Introduction ----------------
    d.h("1. Introduction", 1)
    d.p("La production d'énergie dépend du calendrier (heure, saison) et, pour le solaire et l'éolien, "
        "de la météo. Prévoir cette production aide à équilibrer le réseau et à planifier la production. "
        "Le but de ce projet est de comparer avec rigueur plusieurs techniques de forecasting sur deux "
        "séries de production, puis de dire quelle méthode convient le mieux à chaque problème.")
    d.p("Le rapport suit le plan demandé : analyse exploratoire (EDA), mise en œuvre des quatre modèles, "
        "évaluation (MAE, RMSE, R², R² ajusté, p-values), puis analyse comparative.")

    # ---------------- 2. Méthodologie ----------------
    d.h("2. Méthodologie", 1)
    d.h("2.1 Découpage chronologique", 2)
    d.p("Les observations restent dans l'ordre du temps. Il n'y a aucun mélange aléatoire. "
        "Le test n'est jamais utilisé pour entraîner, choisir les hyperparamètres ou choisir le modèle.")
    rows = []
    for D in R:
        i = D["info"]
        n_tr = None
        rows.append([D["title"], i["n_obs"], f"{i['train_period'][0][:10]} → {i['train_period'][1][:10]}",
                     f"{i['val_period'][0][:10]} → {i['val_period'][1][:10]}",
                     f"{i['test_period'][0][:10]} → {i['test_period'][1][:10]}"])
    d.table(["Jeu", "N", "Train (70 %)", "Validation (15 %)", "Test (15 %)"], rows,
            "Périodes de chaque sous-ensemble", widths=[4.2, 1.4, 3.6, 3.6, 3.6], size=8.5)
    d.h("2.2 Protocole de prévision", 2)
    d.bullets([
        "Horizon h : " + " ; ".join(f"{D['title']} = {D['horizon']} {'mois' if D['freq'] == 'MS' else 'pas de temps'}"
                                    for D in R) + ".",
        "Rolling origin : à chaque origine, le modèle prévoit les h pas suivants, puis l'origine avance de h pas. "
        "Les valeurs observées jusqu'à l'origine sont connues. Les paramètres ne sont pas ré-estimés.",
        "Choix des hyperparamètres : chaque candidat est entraîné sur le train et noté (RMSE) sur la validation.",
        "Modèle final : les hyperparamètres retenus sont ré-ajustés sur train + validation, puis le test est prédit une seule fois.",
        "Prophet ne regarde que la date (et les régresseurs). Il n'exploite donc pas les valeurs récentes, ce qui le désavantage sur les courts horizons.",
        "Variables explicatives (météo) : elles sont supposées connues sur l'horizon (météo « parfaite »). C'est une hypothèse favorable, à garder en tête.",
        "Référence : un modèle naïf saisonnier (valeur de la même heure la veille, ou du même mois l'année précédente) sert de point de comparaison.",
    ])
    d.h("2.3 Modèles", 2)
    for mname in MODELS:
        d.p(f"{mname} ({FAMILLE[mname]}). " + MODEL_DESC[mname])
    d.h("2.4 Indicateurs et tests", 2)
    d.bullets([
        "MAE = moyenne des |erreurs|. RMSE = racine de la moyenne des erreurs au carré (pénalise les grosses erreurs).",
        "R² = 1 − SCR/SCT. R² ajusté = 1 − (1 − R²)(n − 1)/(n − p − 1), où p est le nombre de variables explicatives du modèle "
        "(ARIMA : coefficients AR/MA/exogènes ; Prophet : tendance + termes de Fourier + régresseurs ; XGBoost : variables d'entrée ; "
        "LSTM : variables d'entrée par pas de temps). Pour des modèles non linéaires, ce p est une convention : le R² ajusté sert surtout à comparer.",
        "p-values : on régresse le réel sur le prévu (réel = a + b × prévu) avec des écarts-types robustes HAC (Newey-West). "
        "p_pente teste b = 0. p_MZ (test de Mincer-Zarnowitz) teste a = 0 et b = 1 : prévision sans biais et efficace.",
        "Test de Ljung-Box sur les résidus : teste l'absence d'autocorrélation des erreurs.",
        "Test de Diebold-Mariano (correction de Harvey-Leybourne-Newbold) : teste si deux modèles ont la même précision quadratique.",
        "Intervalles de prédiction à 95 % : fournis par ARIMA et Prophet. Pour XGBoost et le LSTM, on prend ± le quantile 95 % des erreurs absolues de validation. "
        "On mesure la couverture (part des valeurs réelles dans l'intervalle) et la largeur moyenne.",
        "nRMSE = RMSE / moyenne du test. Il sert à comparer des jeux d'unités différentes.",
    ])
    if fast:
        d.p("Remarque : ce rapport vient d'une exécution en mode rapide. Les grilles d'hyperparamètres sont réduites.",
            italic=True, color="C00000")

    # ---------------- 3 et 4. Jeux de données ----------------
    for ix, D in enumerate(R):
        d.page_break()
        d.h(f"{3 + ix}. Jeu de données {ix + 1} : {D['title']}", 1)
        key = D["key"]
        d.h(f"{3 + ix}.1 Analyse exploratoire (EDA)", 2)
        d.p(txt_data(D))
        i = D["info"]
        d.table(["Caractéristique", "Valeur"], [
            ["Nombre d'observations", i["n_obs"]],
            ["Fréquence", i["freq_label"]],
            ["Période couverte", f"{i['start'][:10]} → {i['end'][:10]}"],
            ["Variable cible", f"production ({D['unit']})"],
            ["Variables disponibles", ", ".join(i["variables"])],
            ["Variables explicatives utilisées", ", ".join(D["exog_cols"]) if D["exog_cols"] else "aucune (série seule)"],
            ["Dates manquantes (grille régulière)", i["n_missing_timestamps"]],
            ["Valeurs interpolées (trous courts)", i["n_interpolated"]],
            ["Valeurs non utilisées (trous longs)", i["n_invalid_target"]],
        ], "Caractéristiques du jeu de données", widths=[6.5, 9.5])
        st = D["eda"]["stats"]
        d.table(["Moyenne", "Écart-type", "Min", "Q1", "Médiane", "Q3", "Max", "Asymétrie", "Kurtosis"],
                [[fr(st[k], 2) for k in ("moyenne", "ecart_type", "min", "q25", "mediane", "q75", "max", "asymetrie", "kurtosis")]],
                f"Statistiques descriptives de la production ({D['unit']})", size=8.5, align_first_left=False)
        figs = {Path(p).name: (p, c) for p, c in D["eda"]["figures"]}
        paras = txt_eda(D)
        for name, note in [("eda_serie.png", paras[0]), ("eda_zoom.png", None), ("eda_saison.png", None)]:
            if name in figs:
                d.figure(*figs[name])
                if note:
                    d.p(note)
        if len(paras) > 1:
            d.p(" ".join(paras[1:3]))
        for name in ("eda_stl.png", "eda_acf_pacf.png", "eda_distributions.png", "eda_mois.png",
                     "eda_corr.png", "eda_anomalies.png"):
            if name in figs:
                d.figure(*figs[name])
        d.p(" ".join(paras[3:]))
        d.h("Prétraitements", 3)
        d.bullets([
            "Remise sur une grille de dates régulière.",
            "Interpolation linéaire dans le temps des trous courts. Les trous longs sont exclus de l'entraînement et de l'évaluation.",
            "Valeurs négatives ramenées à 0 (si la production ne peut pas être négative).",
            "Variables explicatives constantes retirées. Variables standardisées pour ARIMA et le LSTM (moyenne et écart-type du train).",
            "Aucune valeur atypique n'est supprimée.",
        ])

        d.h(f"{3 + ix}.2 Résultats des modèles", 2)
        for v in variants_of(D):
            vr = D["variants"][v]
            names = ok_models(vr)
            d.h(f"Cas : {vname_fr(v)}", 3)
            # hyperparamètres
            d.table(["Modèle", "Hyperparamètres retenus (validation)", "RMSE valid."],
                    [[m, vr["models"][m]["label"], fr(vr["models"][m]["val"]["RMSE"])] for m in names],
                    f"Hyperparamètres retenus ({vname_fr(v)})", widths=[2.6, 10.6, 2.8], size=8.5)
            # validation
            d.table(["Modèle", "MAE", "RMSE", "R²"],
                    [[m, fr(vr["models"][m]["val"]["MAE"]), fr(vr["models"][m]["val"]["RMSE"]), fr(vr["models"][m]["val"]["R2"])]
                     for m in names] + [["Naïf saisonnier", fr(vr["naif"]["val"]["MAE"]), fr(vr["naif"]["val"]["RMSE"]), fr(vr["naif"]["val"]["R2"])]],
                    f"Performances sur la validation ({vname_fr(v)})", widths=[4, 3, 3, 3])
            # test
            rows = []
            for m in names:
                t = vr["models"][m]["test"]
                rows.append([m, fr(t["MAE"]), fr(t["RMSE"]), fr(t["R2"]), fr(t["R2_adj"]), t["p_params"],
                             fp(t["p_pente"]), fp(t["p_MZ"]), fr(t["nRMSE"], 3)])
            nt = vr["naif"]["test"]
            rows.append(["Naïf saisonnier", fr(nt["MAE"]), fr(nt["RMSE"]), fr(nt["R2"]), fr(nt["R2_adj"]), 0,
                         fp(nt["p_pente"]), fp(nt["p_MZ"]), fr(nt["nRMSE"], 3)])
            d.table(["Modèle", "MAE", "RMSE", "R²", "R² ajusté", "p", "p (pente)", "p (MZ)", "nRMSE"], rows,
                    f"Performances sur le jeu de TEST ({vname_fr(v)}). p = nombre de variables explicatives.",
                    widths=[3.2, 1.6, 1.7, 1.6, 1.8, 1.0, 1.9, 1.7, 1.7], size=8.5,
                    bold_rows=[names.index(vr["best_val"])])
            d.p("La ligne en gras est le modèle retenu sur la validation.", italic=True, size=9)
            for t_ in txt_results(D, v):
                d.p(t_)
            # DM
            allm = names + ["Naïf saisonnier"]
            rows = []
            for a in allm:
                rows.append([a] + [("-" if a == b else fp(vr["dm"][a][b]["p"])) for b in allm])
            d.table(["Modèle 1 \\ Modèle 2"] + allm, rows,
                    f"Test de Diebold-Mariano : p-values sur le test ({vname_fr(v)}). p < 0,05 = précision significativement différente.",
                    size=8, widths=[3.2] + [2.5] * len(allm))
            # intervalles
            d.table(["Modèle", "Type d'intervalle", "Couverture (cible 95 %)", "Largeur moyenne"],
                    [[m, vr["models"][m]["pi"]["type"], pct(vr["models"][m]["pi"]["coverage"]), fr(vr["models"][m]["pi"]["width"])]
                     for m in names],
                    f"Intervalles de prédiction à 95 % ({vname_fr(v)})", widths=[3, 5.6, 4, 3.4], size=8.5)
            # figures
            for fn in (f"prev_test_{v}.png", f"nuage_{v}.png", f"intervalles_{v}.png", f"barres_{v}.png"):
                for p_, c_ in vr["figures"]:
                    if Path(p_).name == fn:
                        d.figure(p_, c_)
            d.h("Analyse des erreurs", 3)
            for t_ in txt_errors(D, v):
                d.p(t_)
            for fn in (f"residus_{v}.png", f"erreur_groupe_{v}.png", f"lstm_loss_{v}.png", f"xgb_importance_{v}.png"):
                for p_, c_ in vr["figures"]:
                    if Path(p_).name == fn:
                        d.figure(p_, c_, width=13.5 if "lstm" in fn or "xgb" in fn else 15.8)

    # ---------------- 5. Discussion ----------------
    d.page_break()
    d.h("5. Analyse comparative et discussion", 1)
    d.h("5.1 Meilleur modèle pour chaque jeu de données", 2)
    rows = []
    for D in R:
        for v in variants_of(D):
            vr = D["variants"][v]
            bv, bt = vr["best_val"], vr["best_test"]
            rows.append([D["title"], vname_fr(v), bv, fr(vr["models"][bv]["test"]["RMSE"]), bt,
                         fr(vr["models"][bt]["test"]["RMSE"]), fr(vr["naif"]["test"]["RMSE"])])
    d.table(["Jeu", "Cas", "Retenu (validation)", "RMSE test", "Meilleur sur test", "RMSE test", "RMSE naïf"], rows,
            "Meilleurs modèles par jeu de données et par cas", widths=[3.4, 3.2, 2.4, 1.8, 2.4, 1.8, 1.8], size=8)
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        bv = vr["best_val"]
        others = [k for k in ranking(vr) if k != bv]
        sig = [k for k in others if (vr["dm"][bv][k]["p"] is not None and vr["dm"][bv][k]["p"] < 0.05)]
        d.p(f"{D['title']} ({vname_fr(v)}) : {bv} est retenu. Sa différence avec les autres modèles est significative "
            f"(Diebold-Mariano, p < 0,05) pour {liste(sig) if sig else 'aucun d’entre eux'}.")

    d.h("5.2 Statistique, Machine Learning et Deep Learning", 2)
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        m = vr["models"]
        stat = min([k for k in ("ARIMA", "Prophet") if k in ok_models(vr)], key=lambda k: m[k]["test"]["RMSE"])
        parts = [f"meilleur modèle statistique : {stat} (RMSE {fr(m[stat]['test']['RMSE'])})"]
        if "XGBoost" in m and "test" in m["XGBoost"]:
            parts.append(f"Machine Learning : XGBoost ({fr(m['XGBoost']['test']['RMSE'])})")
        if "LSTM" in m and "test" in m["LSTM"]:
            parts.append(f"Deep Learning : LSTM ({fr(m['LSTM']['test']['RMSE'])})")
        fam = {k: FAMILLE[k] for k in ok_models(vr)}
        best = ranking(vr)[0]
        d.p(f"{D['title']} ({vname_fr(v)}) - " + " ; ".join(parts) + f". La meilleure famille sur le test est « {fam[best]} » ({best}).")
    d.p("Ces écarts dépendent de la taille des séries, du réglage et de l'horizon. Une série mensuelle de "
        "quelques centaines de points favorise les modèles statistiques. Une série horaire avec des variables "
        "météo favorise les modèles qui exploitent ces variables.")

    d.h("5.3 Influence de la tendance, de la saisonnalité et des variables explicatives", 2)
    for D in R:
        stl = D["eda"].get("stl", {})
        if "force_tendance" in stl:
            d.p(f"{D['title']} : tendance {level(stl['force_tendance'])} (force {fr(stl['force_tendance'], 2)}), "
                f"saisonnalité {level(stl['force_saison'])} (force {fr(stl['force_saison'], 2)}). "
                + ("Une tendance forte pose problème aux arbres de décision, qui ne prolongent pas une tendance hors de leur plage d'apprentissage. "
                   "C'est pourquoi nous testons aussi une cible en différence (valeur moins valeur d'il y a h pas)." if stl["force_tendance"] >= 0.7 and D["freq"] == "MS"
                   else "La saisonnalité est captée par la structure du modèle (Fourier, saisonnalité Prophet) ou par les retards et le calendrier (XGBoost, LSTM)."))
    for D in R:
        if "avec_exog" in D["variants"] and "sans_exog" in D["variants"]:
            a, b = D["variants"]["sans_exog"], D["variants"]["avec_exog"]
            rows, txt = [], []
            for m in MODELS:
                if m in ok_models(a) and m in ok_models(b):
                    r0, r1 = a["models"][m]["test"]["RMSE"], b["models"][m]["test"]["RMSE"]
                    g = 1 - r1 / r0
                    rows.append([m, fr(r0), fr(r1), pct(g)])
                    txt.append((m, g))
            d.table(["Modèle", "RMSE sans variables", "RMSE avec variables", "Gain relatif"], rows,
                    f"Apport des variables explicatives sur le test - {D['title']}", widths=[3.5, 4, 4, 3.5])
            if txt:
                bestg = max(txt, key=lambda x: x[1])
                worstg = min(txt, key=lambda x: x[1])
                d.p(f"Les variables explicatives ({', '.join(D['exog_cols'])}) changent le RMSE de {pct(worstg[1])} "
                    f"({worstg[0]}) à {pct(bestg[1])} ({bestg[0]}). Un gain positif veut dire que l'erreur diminue.")
            xt = b.get("xgb_top")
            if xt:
                d.p("Variables les plus importantes pour XGBoost : " + liste(f"{k} ({fr(v, 2)})" for k, v in list(xt.items())[:5]) + ".")
            if D["horizon"] > 1:
                d.p("Rappel : la météo est supposée connue sur l'horizon. Dans un cas réel, on utiliserait des prévisions météo, "
                    "ce qui ajouterait de l'incertitude.")

    d.h("5.4 Comparaison des erreurs entre les deux jeux de données", 2)
    rows = []
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        bv = vr["best_val"]
        t = vr["models"][bv]["test"]
        rows.append([D["title"], bv, f"{fr(t['RMSE'])} {D['unit']}", pct(t["nRMSE"]), fr(t["R2"]), fr(vr["naif"]["test"]["R2"])])
    d.table(["Jeu", "Modèle retenu", "RMSE", "nRMSE", "R² test", "R² naïf"], rows,
            "Erreurs du modèle retenu sur chaque jeu", widths=[5, 2.6, 3, 1.8, 1.8, 1.8], size=8.5)
    if len(R) == 2:
        n = [D["variants"][main_variant(D)]["models"][D["variants"][main_variant(D)]["best_val"]]["test"]["nRMSE"] for D in R]
        low = 0 if n[0] < n[1] else 1
        d.p(f"En erreur relative (nRMSE), « {R[low]['title']} » est plus facile à prévoir ({pct(n[low])} contre {pct(n[1 - low])}). "
            "Les deux séries n'ont pas la même fréquence, le même horizon ni la même régularité : cette comparaison est indicative.")

    d.h("5.5 Avantages, limites et coûts de calcul", 2)
    d.table(["Modèle", "Avantages", "Limites"], [[m, AVANTAGES[m][0], AVANTAGES[m][1]] for m in MODELS],
            "Avantages et limites des modèles", widths=[2.2, 7, 7], size=8.5)
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        rows = []
        for m in ok_models(vr):
            t = vr["models"][m]["temps"]
            rows.append([m, fr(t["choix_s"], 1), fr(t["ajustement_s"], 1), fr(t["prediction_s"], 2)])
        d.table(["Modèle", "Choix des hyperparamètres (s)", "Ajustement final (s)", "Prédiction du test (s)"], rows,
                f"Coûts de calcul mesurés - {D['title']} ({vname_fr(v)})", widths=[3, 4.8, 4, 4], size=8.5)
        slow = max(ok_models(vr), key=lambda m: vr["models"][m]["temps"]["choix_s"])
        fast_ = min(ok_models(vr), key=lambda m: vr["models"][m]["temps"]["choix_s"])
        d.p(f"{D['title']} : {slow} est le plus coûteux à régler ({fr(vr['models'][slow]['temps']['choix_s'], 0)} s). "
            f"{fast_} est le moins coûteux ({fr(vr['models'][fast_]['temps']['choix_s'], 0)} s). "
            "Ces temps dépendent de la machine et de la taille des grilles.")

    d.h("5.6 Choix du meilleur modèle pour chaque problème", 2)
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        bv, bt = vr["best_val"], vr["best_test"]
        m = vr["models"]
        s = (f"{D['title']} : nous retenons {bv} ({vname_fr(v)}). Il a la plus faible erreur de validation "
             f"({fr(m[bv]['val']['RMSE'])}), et son RMSE test est de {fr(m[bv]['test']['RMSE'])} {D['unit']} "
             f"contre {fr(vr['naif']['test']['RMSE'])} pour la référence naïve. ")
        if bv != bt:
            s += (f"Sur le test, {bt} est un peu meilleur ({fr(m[bt]['test']['RMSE'])}), mais l'écart n'a pas servi à choisir. ")
        dm = vr["dm"][bv].get(bt) if bt != bv else None
        if dm and dm["p"] is not None:
            s += f"La différence entre {bv} et {bt} est {'significative' if dm['p'] < 0.05 else 'non significative'} (Diebold-Mariano, {peq(dm['p'])}). "
        s += "Points forts : " + AVANTAGES[bv][0] + " Limite principale : " + AVANTAGES[bv][1]
        d.p(s)

    # ---------------- 6. Conclusion ----------------
    d.h("6. Conclusion", 1)
    d.p("Nous avons comparé ARIMA, Prophet, XGBoost et LSTM avec un protocole identique : découpage chronologique 70/15/15, "
        "choix des hyperparamètres sur la validation, évaluation finale sur le test, prévision par origine glissante.")
    for D in R:
        v = main_variant(D)
        vr = D["variants"][v]
        bv = vr["best_val"]
        d.p(f"{D['title']} : {bv} est le modèle retenu (RMSE test {fr(vr['models'][bv]['test']['RMSE'])} {D['unit']}, "
            f"R² {fr(vr['models'][bv]['test']['R2'])}).")
    d.p("Limites : une seule période de test par jeu, une seule graine pour le LSTM, une météo supposée connue, "
        "et des séries parfois courtes (moins de deux ans pour les données horaires, ce qui empêche d'apprendre la saisonnalité annuelle "
        "et crée un changement de saison entre train et test). Pistes : validation croisée en fenêtre glissante, ré-estimation régulière "
        "des modèles, prévisions météo réelles, plusieurs graines pour le LSTM, modèles hybrides.")

    # ---------------- Annexes ----------------
    d.page_break()
    d.h("Annexe A. Résultats de la phase de validation", 1)
    d.p("Pour chaque modèle, les dix meilleurs candidats testés sur la validation (RMSE).")
    for D in R:
        for v in variants_of(D):
            vr = D["variants"][v]
            for m in ok_models(vr):
                g = [x for x in vr["models"][m]["grille"] if x.get("val_RMSE") is not None]
                g = sorted(g, key=lambda x: x["val_RMSE"])[:10]
                d.table(["Candidat", "RMSE validation", "Temps (s)"],
                        [[x["params"], fr(x["val_RMSE"]), fr(x["temps_s"], 1)] for x in g],
                        f"{D['title']} - {m} - {vname_fr(v)}", widths=[10.5, 3, 2.5], size=8)
    d.h("Annexe B. Reproductibilité", 1)
    try:
        import importlib.metadata as im
        vers = {p: im.version(p) for p in ("numpy", "pandas", "statsmodels", "prophet", "xgboost", "keras", "scikit-learn")}
    except Exception:
        vers = {}
    vers = {k: v for k, v in vers.items()}
    d.bullets([f"Python {platform.python_version()} ; " + ", ".join(f"{k} {v}" for k, v in vers.items()),
               f"Graine aléatoire : {config.SEED}. Les résultats du LSTM peuvent varier légèrement d'une machine à l'autre.",
               "Commande : python run_all.py (voir README.md). Les prévisions et les métriques sont enregistrées dans outputs/."])

    path = Path(path) if path else outdir / ("Rapport_Energy_Forecasting_DEMO.docx" if demo else "Rapport_Energy_Forecasting.docx")
    doc.save(path)
    return path
