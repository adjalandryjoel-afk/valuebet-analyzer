"""
═══════════════════════════════════════════════════════
 INTERFACE GRAPHIQUE STREAMLIT

 Lancement : streamlit run app.py
═══════════════════════════════════════════════════════
"""

import os
import sys
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    KellyConfig, ValueBetConfig, PoissonConfig,
    OCRConfig, Paths, SUPPORTED_LEAGUES
)
from modules.ocr_extractor import BetclicScreenshotExtractor, merge_matches
from modules.team_matcher import TeamMatcher
from modules.data_collector import DataCollector
from modules.poisson_model import PoissonPredictor
from modules.elo_rating import EloRatingSystem
from modules.value_detector import ValueBetDetector, MatchAnalysis
from modules.kelly_criterion import KellyStakeCalculator
from modules.database_manager import DatabaseManager
from modules.backtester import Backtester
from modules.match_intel import get_match_intelligence


# ══════════════════════════════════════════════════════
#  CONFIGURATION STREAMLIT
# ══════════════════════════════════════════════════════

st.set_page_config(
    page_title="Value Bet Analyzer — Betclic CI",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Couleurs du thème (alignées sur .streamlit/config.toml)
COLOR_GREEN = "#34D399"
COLOR_RED = "#F87171"


def page_header(icon: str, title: str, caption: str = ""):
    """En-tête de page uniforme : icône Material + titre + sous-titre."""

    st.title(f":material/{icon}: {title}")
    if caption:
        st.caption(caption)
    st.space("small")


# ══════════════════════════════════════════════════════
#  INITIALISATION DES MODULES (avec cache Streamlit)
# ══════════════════════════════════════════════════════

@st.cache_resource
def init_modules():
    """Initialise tous les modules (appelé une seule fois)."""
    return {
        "team_matcher": TeamMatcher(),
        "data_collector": DataCollector(),
        "poisson": PoissonPredictor(),
        "elo": EloRatingSystem(),
        "value_detector": ValueBetDetector(),
        "db": DatabaseManager(),
        "backtester": Backtester(),
        "intel": get_match_intelligence(),
    }


def get_kelly_calculator(bankroll):
    return KellyStakeCalculator(bankroll=bankroll)


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════

def render_sidebar_settings():
    """Réglages dans la barre latérale (sous la navigation)."""

    with st.sidebar:
        st.caption("Betclic Côte d'Ivoire · v3.0 · 25 marchés")

        st.subheader(":material/account_balance_wallet: Bankroll")
        bankroll = st.number_input(
            "Montant (FCFA)",
            min_value=10000,
            max_value=10000000,
            value=int(KellyConfig.DEFAULT_BANKROLL),
            step=10000,
            format="%d"
        )

        st.subheader(":material/tune: Seuils")
        min_value = st.slider(
            "Value minimum (%)",
            min_value=1, max_value=30,
            value=int(ValueBetConfig.MIN_VALUE_THRESHOLD * 100)
        )

        min_confidence = st.slider(
            "Confiance minimum",
            min_value=30, max_value=90,
            value=int(ValueBetConfig.MIN_CONFIDENCE_SCORE)
        )

        st.subheader(":material/groups: Agents d'analyse")
        deep_analysis = st.toggle(
            "Analyse approfondie",
            value=True,
            help="H2H, forme détaillée, contexte et verdict du chef "
                 "analyste. Ajoute quelques secondes par match."
        )

        # Performance globale
        db = init_modules()["db"]
        stats = db.get_performance_stats()

        if stats.get("total_bets", 0) > 0:
            st.space("small")
            st.metric(
                "Paris enregistrés", stats["total_bets"], border=True
            )
            st.metric(
                "ROI global",
                f"{stats.get('roi', 0):.1f}%",
                delta=f"{stats.get('total_profit', 0):+,.0f} FCFA",
                border=True,
            )

        return bankroll, min_value / 100, min_confidence, deep_analysis


# ══════════════════════════════════════════════════════
#  PAGE : UPLOAD DE CAPTURES D'ÉCRAN
# ══════════════════════════════════════════════════════

def page_upload_screenshots(bankroll, min_value, min_confidence,
                            deep_analysis=True):
    """Page d'upload et d'analyse des captures d'écran."""

    page_header(
        "photo_camera", "Upload des captures Betclic",
        "Dépose tes captures d'écran — les matchs, cotes et marchés sont "
        "extraits et fusionnés automatiquement."
    )

    # Upload
    uploaded_files = st.file_uploader(
        "Déposez vos captures d'écran ici",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Minimum 1 capture, maximum 30"
    )

    if uploaded_files:
        st.caption(f":material/check_circle: {len(uploaded_files)} fichier(s) chargé(s)")

        # Afficher les miniatures
        cols = st.columns(min(len(uploaded_files), 5))
        for i, file in enumerate(uploaded_files[:5]):
            with cols[i]:
                img = Image.open(file)
                st.image(img, caption=file.name, width="stretch")

        if len(uploaded_files) > 5:
            st.caption(f"... et {len(uploaded_files) - 5} autre(s)")

        user_notes = st.text_area(
            "Actus des matchs (optionnel)",
            placeholder="Blessés, suspensions, enjeu, turnover... "
                        "Ex : « PSG sans Mbappé (blessé). Finale de coupe "
                        "pour Marseille. »",
            help="L'agent contexte structure ces informations et ajuste "
                 "le modèle en conséquence. Il n'utilise QUE ce que tu "
                 "écris ici — rien n'est inventé."
        )

        # Bouton d'analyse
        if st.button("Lancer l'analyse", type="primary",
                     icon=":material/rocket_launch:", width="stretch"):

            # Sauvegarder les images temporairement
            os.makedirs(OCRConfig.SCREENSHOTS_DIR, exist_ok=True)

            saved_paths = []
            for file in uploaded_files:
                path = os.path.join(OCRConfig.SCREENSHOTS_DIR, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(path)

            # Extraction et analyse
            with st.spinner("🔍 Extraction des données en cours..."):
                extractor = BetclicScreenshotExtractor()
                all_matches = []

                progress_bar = st.progress(0)

                for i, path in enumerate(saved_paths):
                    result = extractor.extract_from_image(path)
                    if result.get("matchs"):
                        all_matches.extend(result["matchs"])
                    progress_bar.progress((i + 1) / len(saved_paths))

                progress_bar.empty()

            all_matches = merge_matches(all_matches)

            if all_matches:
                st.success(f"{len(all_matches)} match(s) extraits !",
                           icon=":material/check_circle:")

                # Analyser les matchs
                analyze_matches_ui(all_matches, bankroll, min_value,
                                   min_confidence, user_notes=user_notes,
                                   deep=deep_analysis)
            else:
                st.error("Aucun match détecté dans les captures.",
                         icon=":material/error:")


# ══════════════════════════════════════════════════════
#  PAGE : SAISIE MANUELLE
# ══════════════════════════════════════════════════════

def page_manual_entry(bankroll, min_value, min_confidence,
                      deep_analysis=True):
    """Page de saisie manuelle des matchs et cotes."""

    page_header(
        "edit_note", "Saisie manuelle des matchs",
        "Entre les cotes Betclic à la main — seuls le 1X2 est obligatoire, "
        "chaque marché ajouté élargit la recherche de value."
    )

    # Nombre de matchs à saisir
    num_matches = st.number_input(
        "Nombre de matchs à analyser",
        min_value=1, max_value=30, value=3
    )

    all_matches = []

    for i in range(num_matches):
        st.subheader(f":material/sports_soccer: Match {i+1}")

        col1, col2 = st.columns(2)

        with col1:
            home = st.text_input(
                "Équipe domicile",
                key=f"home_{i}",
                placeholder="Ex: PSG"
            )

        with col2:
            away = st.text_input(
                "Équipe extérieur",
                key=f"away_{i}",
                placeholder="Ex: Marseille"
            )

        competition = st.text_input(
            "Compétition",
            key=f"comp_{i}",
            placeholder="Ex: Ligue 1 France"
        )

        st.caption("📊 Cotes Betclic :")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            odds_1 = st.number_input("Cote 1 (Dom)", min_value=1.01,
                                      value=1.85, step=0.05, key=f"o1_{i}")
        with col_b:
            odds_x = st.number_input("Cote X (Nul)", min_value=1.01,
                                      value=3.40, step=0.05, key=f"ox_{i}")
        with col_c:
            odds_2 = st.number_input("Cote 2 (Ext)", min_value=1.01,
                                      value=4.20, step=0.05, key=f"o2_{i}")

        # Marchés supplémentaires (expansible)
        extras_input = {}

        with st.expander("Buts du match et BTTS (optionnel)",
                         icon=":material/sports_soccer:"):
            col_d, col_e = st.columns(2)
            with col_d:
                extras_input["over_2_5"] = st.number_input(
                    "Over 2.5", min_value=0.0,
                    value=0.0, step=0.05, key=f"ov_{i}")
                extras_input["btts_oui"] = st.number_input(
                    "BTTS Oui", min_value=0.0,
                    value=0.0, step=0.05, key=f"by_{i}")
            with col_e:
                extras_input["under_2_5"] = st.number_input(
                    "Under 2.5", min_value=0.0,
                    value=0.0, step=0.05, key=f"un_{i}")
                extras_input["btts_non"] = st.number_input(
                    "BTTS Non", min_value=0.0,
                    value=0.0, step=0.05, key=f"bn_{i}")

        with st.expander("Buts par équipe (optionnel)",
                         icon=":material/groups:"):
            st.caption("Marché Betclic « Nombre de buts de l'équipe »")
            col_h, col_a2 = st.columns(2)
            for side, label, col in (("home", "Domicile", col_h),
                                     ("away", "Extérieur", col_a2)):
                with col:
                    st.markdown(f"**{label}**")
                    for line in ("0_5", "1_5", "2_5"):
                        line_txt = line.replace("_", ".")
                        extras_input[f"{side}_over_{line}"] = st.number_input(
                            f"+ de {line_txt}", min_value=0.0, value=0.0,
                            step=0.05, key=f"{side}o{line}_{i}")
                        extras_input[f"{side}_under_{line}"] = st.number_input(
                            f"- de {line_txt}", min_value=0.0, value=0.0,
                            step=0.05, key=f"{side}u{line}_{i}")

        with st.expander("Buts par mi-temps (optionnel)",
                         icon=":material/timer:"):
            st.caption("Nombre de buts du match en 1ère / 2ème mi-temps")
            col_h1, col_h2 = st.columns(2)
            for half, label, col in (("h1", "1ère mi-temps", col_h1),
                                     ("h2", "2ème mi-temps", col_h2)):
                with col:
                    st.markdown(f"**{label}**")
                    for line in ("0_5", "1_5"):
                        line_txt = line.replace("_", ".")
                        extras_input[f"{half}_over_{line}"] = st.number_input(
                            f"+ de {line_txt}", min_value=0.0, value=0.0,
                            step=0.05, key=f"{half}o{line}_{i}")
                        extras_input[f"{half}_under_{line}"] = st.number_input(
                            f"- de {line_txt}", min_value=0.0, value=0.0,
                            step=0.05, key=f"{half}u{line}_{i}")

        with st.expander("Tirs cadrés par équipe (optionnel)",
                         icon=":material/gps_fixed:"):
            st.caption("Choisis la ligne affichée par Betclic (ex. 3.5)")
            col_sh, col_sa = st.columns(2)
            for side, label, col in (("home", "Domicile", col_sh),
                                     ("away", "Extérieur", col_sa)):
                with col:
                    st.markdown(f"**{label}**")
                    sot_line = st.selectbox(
                        "Ligne", ["2.5", "3.5", "4.5", "5.5", "6.5"],
                        index=1, key=f"sotl{side}_{i}")
                    line_key = sot_line.replace(".", "_")
                    extras_input[f"sot_{side}_over_{line_key}"] = st.number_input(
                        f"+ de {sot_line} tirs cadrés", min_value=0.0,
                        value=0.0, step=0.05, key=f"soto{side}_{i}")
                    extras_input[f"sot_{side}_under_{line_key}"] = st.number_input(
                        f"- de {sot_line} tirs cadrés", min_value=0.0,
                        value=0.0, step=0.05, key=f"sotu{side}_{i}")

        if home and away:
            match_data = {
                "equipe_domicile": home,
                "equipe_exterieur": away,
                "competition": competition,
                "cotes": {"1": odds_1, "X": odds_x, "2": odds_2},
                "marches_supplementaires": {
                    k: v for k, v in extras_input.items() if v > 0
                }
            }

            all_matches.append(match_data)

        st.divider()

    # Bouton d'analyse
    if all_matches:
        user_notes = st.text_area(
            "Actus des matchs (optionnel)",
            placeholder="Blessés, suspensions, enjeu, turnover...",
            help="L'agent contexte structure ces informations et ajuste "
                 "le modèle. Il n'utilise QUE ce que tu écris ici."
        )

        if st.button("Analyser tous les matchs", type="primary",
                     icon=":material/rocket_launch:", width="stretch"):
            analyze_matches_ui(all_matches, bankroll, min_value,
                               min_confidence, user_notes=user_notes,
                               deep=deep_analysis)


# ══════════════════════════════════════════════════════
#  ANALYSE DES MATCHS (LOGIQUE COMMUNE)
# ══════════════════════════════════════════════════════

def analyze_matches_ui(matches, bankroll, min_value, min_confidence,
                       user_notes="", deep=True):
    """Analyse les matchs et affiche les résultats dans l'interface."""

    modules = init_modules()
    kelly = get_kelly_calculator(bankroll)
    intel = modules["intel"]

    all_analyses = []
    all_value_bets = []

    progress = st.progress(0, text="Analyse en cours...")

    for i, match in enumerate(matches):
        home_raw = match.get("equipe_domicile", "")
        away_raw = match.get("equipe_exterieur", "")
        cotes = match.get("cotes", {})
        extras = match.get("marches_supplementaires", {})
        competition = match.get("competition", "")

        progress.progress(
            (i + 1) / len(matches),
            text=f"🔄 Analyse de {home_raw} vs {away_raw}..."
        )

        # 1. Identifier les équipes
        match_info = modules["team_matcher"].identify_match(home_raw, away_raw)
        home_name = match_info["home"]["official_name"]
        away_name = match_info["away"]["official_name"]

        # 2. Collecter les données
        all_odds = {**cotes, **extras}
        context = modules["data_collector"].collect_match_data(match_info, all_odds)
        context.competition = competition

        # 3. Estimer les ratings Elo
        if cotes.get("1", 0) > 0 and cotes.get("2", 0) > 0:
            modules["elo"].estimate_rating_from_odds(
                home_name, cotes["1"], cotes["2"], is_home=True
            )
            modules["elo"].estimate_rating_from_odds(
                away_name, cotes["2"], cotes["1"], is_home=False
            )

        elo_pred = modules["elo"].predict(home_name, away_name)

        # 4. Conseil d'agents pré-match (H2H, forme, contexte)
        intel_report = None
        if deep:
            try:
                intel_report = intel.analyze(
                    home_name, away_name, context, user_notes=user_notes
                )
            except Exception:
                intel_report = None

        multipliers = (
            (intel_report.lambda_mult_home, intel_report.lambda_mult_away)
            if intel_report else (1.0, 1.0)
        )

        # 5. Prédiction Poisson (ajustée par les agents)
        poisson_pred = modules["poisson"].predict(
            context, lambda_multipliers=multipliers
        )

        # 5. Détecter les value bets
        analysis = modules["value_detector"].analyze_match(
            home_name, away_name, all_odds,
            poisson_pred, elo_pred, competition
        )

        # 6. Calculer les mises
        stakes = kelly.calculate_all_stakes(analysis)

        # 7. Verdict du chef analyste
        verdict = ""
        if deep and intel_report is not None:
            try:
                verdict = intel.synthesize(
                    analysis, intel_report, bankroll=bankroll
                )
            except Exception:
                verdict = ""

        # Stocker
        analysis._poisson = poisson_pred
        analysis._elo = elo_pred
        analysis._stakes = stakes
        analysis._intel = intel_report
        analysis._verdict = verdict

        # 8. Persister dans l'historique (base SQLite)
        try:
            match_id = modules["db"].save_match_analysis({
                "home_team": home_name,
                "away_team": away_name,
                "competition": competition,
                "match_date": None,
                "odds_1": cotes.get("1"),
                "odds_x": cotes.get("X"),
                "odds_2": cotes.get("2"),
                "odds_over25": extras.get("over_2_5"),
                "odds_under25": extras.get("under_2_5"),
                "odds_btts_yes": extras.get("btts_oui"),
                "odds_btts_no": extras.get("btts_non"),
                "model_prob_1": analysis.model_probs.get("1X2", {}).get("1"),
                "model_prob_x": analysis.model_probs.get("1X2", {}).get("X"),
                "model_prob_2": analysis.model_probs.get("1X2", {}).get("2"),
                "model_prob_over25": analysis.model_probs.get(
                    "OU25", {}).get("over"),
                "model_prob_under25": analysis.model_probs.get(
                    "OU25", {}).get("under"),
                "model_prob_btts_yes": analysis.model_probs.get(
                    "BTTS", {}).get("yes"),
                "model_prob_btts_no": analysis.model_probs.get(
                    "BTTS", {}).get("no"),
                "lambda_home": poisson_pred.lambda_home,
                "lambda_away": poisson_pred.lambda_away,
                "elo_home": elo_pred.home_rating,
                "elo_away": elo_pred.away_rating,
                "predicted_score": analysis.predicted_score,
                "predicted_result": analysis.most_likely_result,
                "confidence": analysis.analysis_confidence,
                "bookmaker_margin": analysis.bookmaker_margin,
            })

            for vb in analysis.value_bets:
                modules["db"].save_value_bet(match_id, {
                    "market": vb.market,
                    "selection": vb.selection,
                    "bookmaker_odds": vb.bookmaker_odds,
                    "fair_odds": vb.fair_odds,
                    "model_probability": vb.model_probability,
                    "implied_probability": vb.implied_probability,
                    "value_percentage": vb.value_percentage,
                    "edge": vb.edge,
                    "confidence_score": vb.confidence_score,
                    "value_rating": vb.value_rating,
                    "kelly_stake": vb.kelly_stake,
                    "recommended_stake": vb.recommended_stake,
                })
        except Exception:
            pass  # l'historique ne doit jamais bloquer une analyse
        all_analyses.append(analysis)
        all_value_bets.extend(analysis.value_bets)

    progress.empty()

    # ── Afficher les résultats ──
    display_results(all_analyses, bankroll)


# ══════════════════════════════════════════════════════
#  AFFICHAGE DES RÉSULTATS
# ══════════════════════════════════════════════════════

def _form_badges(sequence):
    """Convertit "VNDVV" en badges colorés markdown."""

    colors = {"V": "green", "N": "gray", "D": "red"}
    return " ".join(
        f":{colors.get(c, 'gray')}-badge[{c}]" for c in sequence
    )


def _render_intel_section(analysis):
    """Affiche le rapport du conseil d'agents d'un match."""

    report = getattr(analysis, "_intel", None)
    verdict = getattr(analysis, "_verdict", "")

    if report is None and not verdict:
        return

    st.space("small")
    st.markdown(":material/groups: **Conseil des agents**")

    if report is not None:
        col_h2h, col_form = st.columns(2)

        with col_h2h:
            with st.container(border=True):
                st.markdown(":material/swords: **Face-à-face**")
                h2h = report.h2h
                if h2h and h2h.get("sample", 0) >= 2:
                    st.markdown(
                        f"**{h2h['team1_wins']}V - {h2h['draws']}N - "
                        f"{h2h['team2_wins']}D** "
                        f"({analysis.home_team} d'abord, "
                        f"{h2h['sample']} confrontations)"
                    )
                    st.caption(
                        f"{h2h['avg_goals']:.2f} buts/match · "
                        f"BTTS {h2h['btts_rate']*100:.0f}%"
                    )
                    for m in h2h.get("matches", [])[:5]:
                        st.caption(
                            f"{m['date']} — {m['home']} {m['score']} "
                            f"{m['away']}"
                        )
                else:
                    st.caption(
                        "Pas de confrontations directes disponibles."
                    )

        with col_form:
            with st.container(border=True):
                st.markdown(":material/timeline: **Forme récente**")
                shown = False
                for label, form in (
                    (analysis.home_team, report.form_home),
                    (analysis.away_team, report.form_away),
                ):
                    if form and form.get("sequence"):
                        st.markdown(
                            f"{label} : {_form_badges(form['sequence'][:8])}"
                        )
                        shown = True
                if not shown:
                    st.caption("Forme détaillée indisponible.")

        if report.adjust_reasons:
            st.caption(
                ":material/tune: Ajustements du modèle : "
                + " · ".join(report.adjust_reasons)
            )

    if verdict:
        with st.container(border=True):
            st.markdown(":material/psychology: **Verdict du chef analyste**")
            st.markdown(verdict)


def display_results(analyses, bankroll):
    """Affiche tous les résultats de l'analyse."""

    # Résumé en haut
    total_matches = len(analyses)
    matches_with_value = sum(1 for a in analyses if a.has_value)
    total_vb = sum(a.total_value_bets for a in analyses)

    st.space("medium")
    st.subheader(":material/summarize: Résumé de l'analyse")

    total_stake = sum(
        vb.recommended_stake
        for a in analyses for vb in a.value_bets
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matchs analysés", total_matches, border=True)
    col2.metric("Matchs avec value", matches_with_value, border=True)
    col3.metric("Value bets trouvés", total_vb, border=True)
    col4.metric(
        "Investissement conseillé",
        f"{total_stake:,.0f} FCFA",
        border=True,
    )

    st.space("small")

    # ── Détail par match ──
    for analysis in analyses:

        exp_icon = (":material/trending_up:" if analysis.has_value
                    else ":material/do_not_disturb_on:")
        vb_note = (f"{analysis.total_value_bets} value bet(s)"
                   if analysis.has_value else "pas de value")

        with st.expander(
            f"{analysis.home_team} vs {analysis.away_team}"
            + (f" — {analysis.competition}" if analysis.competition else "")
            + f" · {vb_note}",
            expanded=analysis.has_value,
            icon=exp_icon,
        ):

            # Probabilités 1X2
            col1, col2 = st.columns([3, 2])

            with col1:
                st.caption("Probabilités modèle vs Betclic")

                probs_1x2 = analysis.model_probs.get("1X2", {})

                # Colonnes numériques → barres de progression et
                # mise en forme via column_config
                data = {
                    "Marché": ["1 (Domicile)", "X (Nul)", "2 (Extérieur)"],
                    "Modèle": [probs_1x2.get(k, 0) * 100
                               for k in ("1", "X", "2")],
                    "Betclic": [
                        (100 / analysis.odds[k])
                        if analysis.odds.get(k, 0) > 0 else None
                        for k in ("1", "X", "2")
                    ],
                    "Cote": [
                        analysis.odds.get(k, 0) or None
                        for k in ("1", "X", "2")
                    ],
                }

                # Ajouter la colonne Value
                values = []
                for key in ["1", "X", "2"]:
                    mp = probs_1x2.get(key, 0)
                    bo = analysis.odds.get(key, 0)
                    if bo > 0 and mp > 0:
                        values.append((bo * mp - 1) * 100)
                    else:
                        values.append(None)

                data["Value"] = values

                # Marchés supplémentaires (Over/Under 2.5 et BTTS)
                probs_ou = analysis.model_probs.get("OU25", {})
                probs_btts = analysis.model_probs.get("BTTS", {})

                extra_rows = [
                    ("Over 2.5", probs_ou.get("over", 0), "over_2_5"),
                    ("Under 2.5", probs_ou.get("under", 0), "under_2_5"),
                    ("BTTS Oui", probs_btts.get("yes", 0), "btts_oui"),
                    ("BTTS Non", probs_btts.get("no", 0), "btts_non"),
                ]

                # Totaux par équipe (home/away over/under 0.5, 1.5, 2.5)
                for side, team, group in (
                    ("home", analysis.home_team, "HOME_TOTALS"),
                    ("away", analysis.away_team, "AWAY_TOTALS"),
                ):
                    probs_tot = analysis.model_probs.get(group, {})
                    for line in ("0_5", "1_5", "2_5"):
                        for ou in ("over", "under"):
                            extra_rows.append((
                                f"{team} {ou.capitalize()} {line.replace('_', '.')}",
                                probs_tot.get(f"{ou}_{line}", 0),
                                f"{side}_{ou}_{line}",
                            ))

                # Buts par mi-temps (1MT / 2MT over/under 0.5, 1.5)
                for half, half_label in (("h1", "1MT"), ("h2", "2MT")):
                    probs_half = analysis.model_probs.get(half.upper(), {})
                    for line in ("0_5", "1_5"):
                        for ou in ("over", "under"):
                            extra_rows.append((
                                f"{half_label} {ou.capitalize()} {line.replace('_', '.')}",
                                probs_half.get(f"{ou}_{line}", 0),
                                f"{half}_{ou}_{line}",
                            ))

                # Tirs cadrés par équipe (lignes variables, déduites des cotes)
                for side, team in (("home", analysis.home_team),
                                   ("away", analysis.away_team)):
                    probs_sot = analysis.model_probs.get(f"SOT_{side.upper()}", {})
                    prefix = f"sot_{side}_"
                    sot_lines = []
                    for odds_key in analysis.odds:
                        if not odds_key.startswith(prefix):
                            continue
                        parts = odds_key[len(prefix):].split("_", 1)
                        if (len(parts) == 2 and parts[0] in ("over", "under")
                                and parts[1].replace("_", "", 1).isdigit()):
                            sot_lines.append((parts[0], parts[1], odds_key))
                    sot_lines.sort(key=lambda s: (float(s[1].replace("_", ".")),
                                                  s[0] != "over"))
                    for ou, line, odds_key in sot_lines:
                        extra_rows.append((
                            f"Tirs cadrés {team} {ou.capitalize()} {line.replace('_', '.')}",
                            probs_sot.get(f"{ou}_{line}", 0),
                            odds_key,
                        ))

                for label, mp, odds_key in extra_rows:
                    bo = float(analysis.odds.get(odds_key, 0) or 0)
                    if bo <= 1:
                        continue
                    data["Marché"].append(label)
                    data["Modèle"].append(mp * 100)
                    data["Betclic"].append(100 / bo)
                    data["Cote"].append(bo)
                    if mp > 0:
                        data["Value"].append((bo * mp - 1) * 100)
                    else:
                        data["Value"].append(None)

                df = pd.DataFrame(data)

                def _value_color(v):
                    if pd.isna(v):
                        return ""
                    if v > 0:
                        return f"color: {COLOR_GREEN}; font-weight: 600"
                    return f"color: {COLOR_RED}"

                st.dataframe(
                    df.style.map(_value_color, subset=["Value"]),
                    hide_index=True,
                    width="stretch",
                    height=min(38 + 35 * len(df), 460),
                    column_config={
                        "Marché": st.column_config.TextColumn(
                            "Marché", width="medium"),
                        "Modèle": st.column_config.ProgressColumn(
                            "Prob. modèle", min_value=0, max_value=100,
                            format="%.1f%%"),
                        "Betclic": st.column_config.NumberColumn(
                            "Prob. Betclic", format="%.1f%%"),
                        "Cote": st.column_config.NumberColumn(
                            "Cote", format="%.2f"),
                        "Value": st.column_config.NumberColumn(
                            "Value", format="%+.1f%%"),
                    },
                )

            with col2:
                # Graphique radar
                probs = probs_1x2
                if probs:
                    fig = go.Figure()

                    categories = ['Domicile', 'Nul', 'Extérieur']
                    model_vals = [
                        probs.get('1', 0)*100,
                        probs.get('X', 0)*100,
                        probs.get('2', 0)*100,
                    ]

                    betclic_vals = []
                    for key in ['1', 'X', '2']:
                        o = analysis.odds.get(key, 0)
                        betclic_vals.append(100/o if o > 0 else 0)

                    fig.add_trace(go.Scatterpolar(
                        r=model_vals + [model_vals[0]],
                        theta=categories + [categories[0]],
                        name='Modèle',
                        line=dict(color=COLOR_GREEN, width=3),
                        fill='toself',
                        fillcolor='rgba(52, 211, 153, 0.15)'
                    ))

                    fig.add_trace(go.Scatterpolar(
                        r=betclic_vals + [betclic_vals[0]],
                        theta=categories + [categories[0]],
                        name='Betclic',
                        line=dict(color=COLOR_RED, width=3),
                        fill='toself',
                        fillcolor='rgba(248, 113, 113, 0.15)'
                    ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, range=[0, 70],
                                gridcolor='#334155',
                            ),
                            angularaxis=dict(gridcolor='#334155'),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        showlegend=True,
                        legend=dict(orientation="h", y=-0.1),
                        height=320,
                        margin=dict(l=40, r=40, t=30, b=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#F1F5F9', family='Inter')
                    )

                    st.plotly_chart(fig)

            # Value bets
            if analysis.value_bets:
                st.markdown(
                    f":material/target: **{analysis.total_value_bets} "
                    f"value bet(s) détecté(s)**"
                )

                for vb in analysis.value_bets:
                    with st.container(border=True):
                        c_sel, c_conf, c_mise, c_pct = st.columns(
                            [3, 1, 1.3, 1], vertical_alignment="center"
                        )
                        with c_sel:
                            st.markdown(
                                f"**{vb.market} → {vb.selection}** "
                                f"@ **{vb.bookmaker_odds:.2f}**  \n"
                                f":green-badge[:material/trending_up: "
                                f"+{vb.value_percentage*100:.1f}%] "
                                f"{vb.value_rating}"
                            )
                        c_conf.metric(
                            "Confiance", f"{vb.confidence_score:.0f}/100")
                        c_mise.metric(
                            "Mise", f"{vb.recommended_stake:,.0f} FCFA")
                        c_pct.metric(
                            "Bankroll", f"{vb.kelly_stake:.1f}%")
            else:
                st.caption(
                    ":material/do_not_disturb_on: Aucun value bet détecté — "
                    "les cotes semblent correctement calibrées sur ce match."
                )

            # Infos additionnelles
            st.space("small")
            st.markdown(
                f":material/scoreboard: Score prédit : **{analysis.predicted_score}** · "
                f":material/emoji_events: **{analysis.most_likely_result}** · "
                f":material/percent: Marge Betclic : **{analysis.bookmaker_margin:.1f}%** · "
                f":material/verified: Confiance : **{analysis.analysis_confidence:.0f}/100**"
            )

            if analysis.data_warning:
                st.warning(analysis.data_warning, icon=":material/warning:")

            _render_intel_section(analysis)

    # ── Tableau récapitulatif final ──
    all_vbs = []
    for a in analyses:
        for vb in a.value_bets:
            all_vbs.append({
                "Match": f"{a.home_team} vs {a.away_team}",
                "Marché": vb.market.split(" - ")[-1] if " - " in vb.market else vb.market,
                "Sélection": vb.selection,
                "Cote": vb.bookmaker_odds,
                "Value": vb.value_percentage * 100,
                "Rating": vb.value_rating,
                "Confiance": vb.confidence_score,
                "Mise (FCFA)": vb.recommended_stake,
            })

    if all_vbs:
        st.space("medium")
        st.subheader(":material/emoji_events: Tous les value bets recommandés")

        df_vbs = pd.DataFrame(all_vbs).sort_values(
            "Value", ascending=False
        )

        def _vb_color(v):
            return f"color: {COLOR_GREEN}; font-weight: 600"

        st.dataframe(
            df_vbs.style.map(_vb_color, subset=["Value"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Cote": st.column_config.NumberColumn(format="%.2f"),
                "Value": st.column_config.NumberColumn(format="%+.1f%%"),
                "Confiance": st.column_config.ProgressColumn(
                    "Confiance", min_value=0, max_value=100, format="%.0f"),
                "Mise (FCFA)": st.column_config.NumberColumn(
                    format="localized"),
            },
        )


# ══════════════════════════════════════════════════════
#  PAGE : HISTORIQUE DES ANALYSES
# ══════════════════════════════════════════════════════

def page_history():
    """Historique de toutes les analyses effectuées."""

    page_header(
        "receipt_long", "Historique des analyses",
        "Chaque analyse (captures ou saisie manuelle) est archivée ici "
        "avec sa prédiction et ses value bets."
    )

    db = init_modules()["db"]

    search = st.text_input(
        "Rechercher",
        placeholder="Filtrer par équipe ou compétition...",
        label_visibility="collapsed",
        icon=":material/search:",
    )

    history = db.get_analysis_history(search=search.strip())

    if not history:
        st.info(
            "Aucune analyse archivée pour l'instant — lance ta première "
            "analyse depuis Upload captures ou Saisie manuelle.",
            icon=":material/info:",
        )
        return

    # Métriques de synthèse
    n_total = len(history)
    n_with_value = sum(1 for h in history if (h["n_value_bets"] or 0) > 0)
    n_vbs = sum(h["n_value_bets"] or 0 for h in history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Analyses", n_total, border=True)
    c2.metric("Matchs avec value", n_with_value, border=True)
    c3.metric("Value bets détectés", n_vbs, border=True)

    st.space("small")
    st.caption(
        ":material/touch_app: Clique sur une ligne pour voir le détail. "
        "Sur l'app en ligne, l'historique repart de zéro à chaque "
        "redémarrage du serveur — l'historique complet vit sur ton PC."
    )

    df_hist = pd.DataFrame([{
        "id": h["id"],
        "Date": (h["analysis_date"] or "")[:16],
        "Match": f"{h['home_team']} vs {h['away_team']}",
        "Compétition": h["competition"] or "—",
        "Prédiction": (f"{h['predicted_score'] or '—'} · "
                       f"{h['predicted_result'] or '—'}"),
        "Confiance": h["confidence"] or 0,
        "Value bets": h["n_value_bets"] or 0,
        "Meilleure value": ((h["best_value"] or 0) * 100
                            if h["best_value"] else None),
    } for h in history])

    event = st.dataframe(
        df_hist,
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(df_hist), 420),
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "id": None,
            "Confiance": st.column_config.ProgressColumn(
                "Confiance", min_value=0, max_value=100, format="%.0f"),
            "Meilleure value": st.column_config.NumberColumn(
                "Meilleure value", format="%+.1f%%"),
        },
    )

    # Détail de l'analyse sélectionnée
    rows = event.selection.rows if event and event.selection else []
    if not rows:
        return

    selected = history[rows[0]]
    st.space("small")

    with st.container(border=True):
        st.markdown(
            f":material/sports_soccer: **{selected['home_team']} vs "
            f"{selected['away_team']}**"
            + (f" — {selected['competition']}"
               if selected['competition'] else "")
        )
        st.caption(
            f"Analysé le {(selected['analysis_date'] or '')[:16]} · "
            f"Cotes 1X2 : {selected['odds_1'] or '—'} / "
            f"{selected['odds_x'] or '—'} / {selected['odds_2'] or '—'} · "
            f"Marge : {selected['bookmaker_margin'] or 0:.1f}%"
        )

        vbs = db.get_value_bets_for_match(selected["id"])

        if vbs:
            statut = {None: "⏳ En attente", "win": "✅ Gagné",
                      "loss": "❌ Perdu", "void": "➖ Non joué"}
            df_vbs = pd.DataFrame([{
                "Marché": vb["market"],
                "Sélection": vb["selection"],
                "Cote": vb["bookmaker_odds"],
                "Value": (vb["value_percentage"] or 0) * 100,
                "Mise (FCFA)": vb["recommended_stake"] or 0,
                "Statut": statut.get(vb["result"], vb["result"]),
            } for vb in vbs])

            st.dataframe(
                df_vbs,
                hide_index=True,
                width="stretch",
                column_config={
                    "Cote": st.column_config.NumberColumn(format="%.2f"),
                    "Value": st.column_config.NumberColumn(
                        format="%+.1f%%"),
                    "Mise (FCFA)": st.column_config.NumberColumn(
                        format="localized"),
                },
            )
        else:
            st.caption(
                ":material/do_not_disturb_on: Aucun value bet détecté "
                "sur cette analyse."
            )


# ══════════════════════════════════════════════════════
#  PAGE : TABLEAU DE BORD
# ══════════════════════════════════════════════════════

def page_dashboard():
    """Tableau de bord avec les statistiques globales."""

    page_header(
        "monitoring", "Tableau de bord",
        "Suivi de tes performances réelles : paris, ROI et bankroll."
    )

    db = init_modules()["db"]
    stats = db.get_performance_stats()

    if not stats or stats.get("total_bets", 0) == 0:
        st.info(
            "Aucune donnée pour l'instant : analyse des matchs, enregistre "
            "tes paris, puis renseigne leurs résultats dans Backtesting.",
            icon=":material/info:",
        )
        return

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total paris", stats.get("total_bets", 0), border=True)

    with col2:
        wr = stats.get("win_rate", 0)
        st.metric("Taux de réussite", f"{wr:.1f}%", border=True)

    with col3:
        roi = stats.get("roi", 0)
        st.metric(
            "ROI",
            f"{roi:+.1f}%",
            delta=f"{stats.get('total_profit', 0):+,.0f} FCFA",
            border=True,
        )

    with col4:
        st.metric(
            "Value moyenne",
            f"{stats.get('avg_value', 0)*100:+.1f}%",
            border=True,
        )

    # Graphique d'évolution du bankroll
    backtester = init_modules()["backtester"]
    result = backtester.run_backtest()

    if result.bankroll_history:
        st.space("small")
        st.subheader(":material/finance_mode: Évolution du bankroll")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=result.bankroll_history,
            mode='lines',
            name='Bankroll',
            line=dict(color=COLOR_GREEN, width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 211, 153, 0.08)'
        ))

        fig.update_layout(
            height=400,
            xaxis_title="Nombre de paris",
            yaxis_title="Bankroll (FCFA)",
            margin=dict(l=40, r=40, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F1F5F9', family='Inter'),
            xaxis=dict(gridcolor='#334155'),
            yaxis=dict(gridcolor='#334155'),
        )

        st.plotly_chart(fig)


# ══════════════════════════════════════════════════════
#  PAGE : BACKTESTING
# ══════════════════════════════════════════════════════

def page_backtesting():
    """Page de backtesting et validation."""

    page_header(
        "history", "Backtesting & validation",
        "Renseigne les résultats de tes paris pour mesurer la vraie "
        "performance du modèle."
    )

    backtester = init_modules()["backtester"]

    # Charger les paris en attente
    db = init_modules()["db"]
    pending = db.get_pending_bets()

    if pending:
        st.subheader(":material/pending: Paris en attente de résultat")

        for bet in pending:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.write(
                    f"**{bet['home_team']} vs {bet['away_team']}** — "
                    f"{bet['market']} → {bet['selection']} @ {bet['bookmaker_odds']}"
                )

            with col2:
                if st.button("Gagné", key=f"win_{bet['id']}",
                             icon=":material/check_circle:"):
                    profit = bet['recommended_stake'] * (bet['bookmaker_odds'] - 1)
                    db.update_bet_result(bet['id'], "win", profit)
                    st.rerun()

            with col3:
                if st.button("Perdu", key=f"loss_{bet['id']}",
                             icon=":material/cancel:"):
                    db.update_bet_result(bet['id'], "loss", -bet['recommended_stake'])
                    st.rerun()

            with col4:
                if st.button("Non joué", key=f"void_{bet['id']}",
                             icon=":material/remove_circle_outline:"):
                    db.update_bet_result(bet['id'], "void", 0.0)
                    st.rerun()

    # Lancer le backtest
    st.space("small")

    if st.button("Recalculer le backtest", type="primary",
                 icon=":material/refresh:"):
        result = backtester.run_backtest()

        if result.total_bets > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("ROI", f"{result.roi:+.1f}%", border=True)
            col2.metric("Max drawdown", f"{result.max_drawdown:.1f}%",
                        border=True)
            col3.metric(
                "P-value",
                f"{result.p_value:.4f}",
                delta="Significatif" if result.is_significant else "Non significatif",
                border=True,
            )
        else:
            st.info("Aucun pari résolu disponible pour le backtest.",
                    icon=":material/info:")


# ══════════════════════════════════════════════════════
#  PAGE : PARAMÈTRES
# ══════════════════════════════════════════════════════

def page_settings():
    """Page de configuration des paramètres."""

    page_header(
        "settings", "Paramètres",
        "Réglages des modèles et des clés API."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        ":material/casino: Modèle Poisson",
        ":material/leaderboard: Elo",
        ":material/target: Value bet",
        ":material/key: APIs"
    ])

    with tab1:
        st.subheader("Paramètres du modèle de Poisson")

        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Score max dans la matrice",
                value=PoissonConfig.MAX_GOALS,
                min_value=3, max_value=10
            )
            st.number_input(
                "Avantage domicile (%)",
                value=int(PoissonConfig.HOME_ADVANTAGE * 100 - 100),
                min_value=0, max_value=20
            )
        with col2:
            st.number_input(
                "Matchs récents considérés",
                value=PoissonConfig.RECENT_MATCHES_COUNT,
                min_value=3, max_value=20
            )
            st.slider(
                "Poids forme récente",
                value=PoissonConfig.RECENT_FORM_WEIGHT,
                min_value=0.0, max_value=1.0, step=0.05
            )

    with tab2:
        st.subheader("Paramètres Elo")

        from config import EloConfig
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Rating initial", value=EloConfig.INITIAL_RATING)
            st.number_input("Facteur K", value=EloConfig.K_FACTOR)

    with tab3:
        st.subheader("Paramètres Value Bet")

        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Value minimum (%)",
                value=int(ValueBetConfig.MIN_VALUE_THRESHOLD * 100)
            )
            st.number_input(
                "Confiance minimum",
                value=int(ValueBetConfig.MIN_CONFIDENCE_SCORE)
            )
        with col2:
            st.number_input("Cote minimum", value=ValueBetConfig.MIN_ODDS, step=0.1)
            st.number_input("Cote maximum", value=ValueBetConfig.MAX_ODDS, step=0.5)

    with tab4:
        st.subheader(":material/key: Clés API")
        st.text_input("OpenAI API Key", type="password", value="sk-...")
        st.text_input("RapidAPI Key", type="password", value="")
        st.text_input("The Odds API Key", type="password", value="")

        if st.button("Sauvegarder", icon=":material/save:"):
            st.success("Paramètres sauvegardés !",
                       icon=":material/check_circle:")


# ══════════════════════════════════════════════════════
#  POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════

def page_home():
    """Page d'accueil."""

    page_header(
        "sports_soccer", "Value Bet Analyzer",
        "Analyse les cotes de Betclic Côte d'Ivoire et détecte les value "
        "bets — les paris où la probabilité réelle dépasse ce que suggère "
        "la cote."
    )

    st.markdown(
        ":green-badge[:material/check: 25 marchés analysés] "
        ":blue-badge[:material/psychology: Poisson + Elo + stats réelles] "
        ":violet-badge[:material/calculate: Mises Kelly]"
    )

    st.space("medium")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown(":material/photo_camera: **Upload de captures**")
            st.caption(
                "Dépose tes captures Betclic : matchs, cotes et marchés "
                "sont lus et fusionnés automatiquement."
            )

    with col2:
        with st.container(border=True):
            st.markdown(":material/edit_note: **Saisie manuelle**")
            st.caption(
                "Entre les cotes à la main, du 1X2 aux tirs cadrés, "
                "quand tu n'as pas de capture sous la main."
            )

    with col3:
        with st.container(border=True):
            st.markdown(":material/monitoring: **Tableau de bord**")
            st.caption(
                "Suis ton ROI réel, ton bankroll et la performance du "
                "modèle sur tes paris enregistrés."
            )

    st.space("medium")
    st.caption(
        ":material/health_and_safety: Les mises proposées sont volontairement "
        "prudentes (quart de Kelly, plafond 5% du bankroll). Ne mise jamais "
        "plus que ce que tu peux te permettre de perdre."
    )


def main():
    """Point d'entrée de l'application Streamlit."""

    bankroll, min_value, min_confidence, deep_analysis = (
        render_sidebar_settings()
    )

    def _upload():
        page_upload_screenshots(bankroll, min_value, min_confidence,
                                deep_analysis)

    def _manual():
        page_manual_entry(bankroll, min_value, min_confidence,
                          deep_analysis)

    pg = st.navigation([
        st.Page(page_home, title="Accueil",
                icon=":material/home:", default=True),
        st.Page(_upload, title="Upload captures",
                icon=":material/photo_camera:", url_path="upload"),
        st.Page(_manual, title="Saisie manuelle",
                icon=":material/edit_note:", url_path="saisie"),
        st.Page(page_dashboard, title="Tableau de bord",
                icon=":material/monitoring:", url_path="dashboard"),
        st.Page(page_history, title="Historique",
                icon=":material/receipt_long:", url_path="historique"),
        st.Page(page_backtesting, title="Backtesting",
                icon=":material/history:", url_path="backtesting"),
        st.Page(page_settings, title="Paramètres",
                icon=":material/settings:", url_path="parametres"),
    ])

    pg.run()


if __name__ == "__main__":
    main()
