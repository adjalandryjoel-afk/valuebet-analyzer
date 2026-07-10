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


# ══════════════════════════════════════════════════════
#  CONFIGURATION STREAMLIT
# ══════════════════════════════════════════════════════

st.set_page_config(
    page_title="⚽ Value Bet Analyzer — Betclic CI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #0a3d0a, #1a5c1a);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
        margin: 1rem 0;
        color: white;
    }
    .no-value-card {
        background: linear-gradient(135deg, #3d0a0a, #5c1a1a);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
        color: white;
    }
    .metric-card {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


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
    }


def get_kelly_calculator(bankroll):
    return KellyStakeCalculator(bankroll=bankroll)


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════

def render_sidebar():
    """Affiche la barre latérale avec les paramètres."""

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/football2--v1.png", width=80)
        st.title("⚽ Value Bet Analyzer")
        st.caption("Betclic Côte d'Ivoire")

        st.divider()

        # Navigation
        page = st.radio(
            "📍 Navigation",
            [
                "🏠 Analyse de Matchs",
                "📸 Upload Captures",
                "✍️ Saisie Manuelle",
                "📊 Tableau de Bord",
                "📈 Backtesting",
                "⚙️ Paramètres"
            ],
            index=0
        )

        st.divider()

        # Paramètres rapides
        st.subheader("💰 Bankroll")
        bankroll = st.number_input(
            "Montant (FCFA)",
            min_value=10000,
            max_value=10000000,
            value=int(KellyConfig.DEFAULT_BANKROLL),
            step=10000,
            format="%d"
        )

        st.subheader("🎯 Seuils")
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

        st.divider()

        # Infos
        db = init_modules()["db"]
        stats = db.get_performance_stats()

        if stats.get("total_bets", 0) > 0:
            st.metric("📊 Paris enregistrés", stats["total_bets"])
            st.metric(
                "📈 ROI Global",
                f"{stats.get('roi', 0):.1f}%",
                delta=f"{stats.get('total_profit', 0):+,.0f} FCFA"
            )

        return page, bankroll, min_value / 100, min_confidence


# ══════════════════════════════════════════════════════
#  PAGE : UPLOAD DE CAPTURES D'ÉCRAN
# ══════════════════════════════════════════════════════

def page_upload_screenshots(bankroll, min_value, min_confidence):
    """Page d'upload et d'analyse des captures d'écran."""

    st.markdown(
        '<div class="main-header">📸 Upload des Captures Betclic</div>',
        unsafe_allow_html=True
    )

    st.info(
        "📌 Envoyez vos captures d'écran de Betclic Côte d'Ivoire. "
        "Le logiciel extraira automatiquement les matchs et les cotes "
        "pour les analyser."
    )

    # Upload
    uploaded_files = st.file_uploader(
        "Déposez vos captures d'écran ici",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Minimum 1 capture, maximum 30"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} fichier(s) chargé(s)")

        # Afficher les miniatures
        cols = st.columns(min(len(uploaded_files), 5))
        for i, file in enumerate(uploaded_files[:5]):
            with cols[i]:
                img = Image.open(file)
                st.image(img, caption=file.name, use_container_width=True)

        if len(uploaded_files) > 5:
            st.caption(f"... et {len(uploaded_files) - 5} autre(s)")

        # Bouton d'analyse
        if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):

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
                st.success(f"✅ {len(all_matches)} match(s) extraits !")

                # Analyser les matchs
                analyze_matches_ui(all_matches, bankroll, min_value, min_confidence)
            else:
                st.error("❌ Aucun match détecté dans les captures.")


# ══════════════════════════════════════════════════════
#  PAGE : SAISIE MANUELLE
# ══════════════════════════════════════════════════════

def page_manual_entry(bankroll, min_value, min_confidence):
    """Page de saisie manuelle des matchs et cotes."""

    st.markdown(
        '<div class="main-header">✍️ Saisie Manuelle des Matchs</div>',
        unsafe_allow_html=True
    )

    # Nombre de matchs à saisir
    num_matches = st.number_input(
        "Nombre de matchs à analyser",
        min_value=1, max_value=30, value=3
    )

    all_matches = []

    for i in range(num_matches):
        st.subheader(f"⚽ Match {i+1}")

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
        with st.expander("📈 Marchés supplémentaires (optionnel)"):
            col_d, col_e = st.columns(2)
            with col_d:
                over25 = st.number_input("Over 2.5", min_value=0.0,
                                          value=0.0, step=0.05, key=f"ov_{i}")
                btts_y = st.number_input("BTTS Oui", min_value=0.0,
                                          value=0.0, step=0.05, key=f"by_{i}")
            with col_e:
                under25 = st.number_input("Under 2.5", min_value=0.0,
                                           value=0.0, step=0.05, key=f"un_{i}")
                btts_n = st.number_input("BTTS Non", min_value=0.0,
                                          value=0.0, step=0.05, key=f"bn_{i}")

        if home and away:
            match_data = {
                "equipe_domicile": home,
                "equipe_exterieur": away,
                "competition": competition,
                "cotes": {"1": odds_1, "X": odds_x, "2": odds_2},
                "marches_supplementaires": {}
            }

            if over25 > 0:
                match_data["marches_supplementaires"]["over_2_5"] = over25
            if under25 > 0:
                match_data["marches_supplementaires"]["under_2_5"] = under25
            if btts_y > 0:
                match_data["marches_supplementaires"]["btts_oui"] = btts_y
            if btts_n > 0:
                match_data["marches_supplementaires"]["btts_non"] = btts_n

            all_matches.append(match_data)

        st.divider()

    # Bouton d'analyse
    if all_matches:
        if st.button("🚀 Analyser tous les matchs", type="primary",
                      use_container_width=True):
            analyze_matches_ui(all_matches, bankroll, min_value, min_confidence)


# ══════════════════════════════════════════════════════
#  ANALYSE DES MATCHS (LOGIQUE COMMUNE)
# ══════════════════════════════════════════════════════

def analyze_matches_ui(matches, bankroll, min_value, min_confidence):
    """Analyse les matchs et affiche les résultats dans l'interface."""

    modules = init_modules()
    kelly = get_kelly_calculator(bankroll)

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

        # 4. Prédiction Poisson
        poisson_pred = modules["poisson"].predict(context)

        # 5. Détecter les value bets
        analysis = modules["value_detector"].analyze_match(
            home_name, away_name, all_odds,
            poisson_pred, elo_pred, competition
        )

        # 6. Calculer les mises
        stakes = kelly.calculate_all_stakes(analysis)

        # Stocker
        analysis._poisson = poisson_pred
        analysis._elo = elo_pred
        analysis._stakes = stakes
        all_analyses.append(analysis)
        all_value_bets.extend(analysis.value_bets)

    progress.empty()

    # ── Afficher les résultats ──
    display_results(all_analyses, bankroll)


# ══════════════════════════════════════════════════════
#  AFFICHAGE DES RÉSULTATS
# ══════════════════════════════════════════════════════

def display_results(analyses, bankroll):
    """Affiche tous les résultats de l'analyse."""

    # Résumé en haut
    total_matches = len(analyses)
    matches_with_value = sum(1 for a in analyses if a.has_value)
    total_vb = sum(a.total_value_bets for a in analyses)

    st.markdown("---")
    st.subheader("📋 Résumé de l'analyse")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚽ Matchs analysés", total_matches)
    col2.metric("✅ Matchs avec value", matches_with_value)
    col3.metric("🎰 Value bets trouvés", total_vb)

    total_stake = sum(
        vb.recommended_stake
        for a in analyses for vb in a.value_bets
    )
    col4.metric(
        "💰 Investissement total",
        f"{total_stake:,.0f} FCFA"
    )

    st.markdown("---")

    # ── Détail par match ──
    for analysis in analyses:

        # Couleur selon la value
        if analysis.has_value:
            icon = "🟢"
        else:
            icon = "🔴"

        with st.expander(
            f"{icon} {analysis.home_team} vs {analysis.away_team} "
            f"— {analysis.competition} "
            f"({'✅ ' + str(analysis.total_value_bets) + ' value bet(s)' if analysis.has_value else '❌ Pas de value'})",
            expanded=analysis.has_value
        ):

            # Probabilités 1X2
            col1, col2 = st.columns([3, 2])

            with col1:
                st.caption("📊 Probabilités Modèle vs Betclic")

                probs_1x2 = analysis.model_probs.get("1X2", {})

                data = {
                    "Résultat": ["1 (Domicile)", "X (Nul)", "2 (Extérieur)"],
                    "Modèle": [
                        f"{probs_1x2.get('1', 0)*100:.1f}%",
                        f"{probs_1x2.get('X', 0)*100:.1f}%",
                        f"{probs_1x2.get('2', 0)*100:.1f}%",
                    ],
                    "Betclic": [
                        f"{1/analysis.odds.get('1', 99)*100:.1f}%" if analysis.odds.get('1', 0) > 0 else "—",
                        f"{1/analysis.odds.get('X', 99)*100:.1f}%" if analysis.odds.get('X', 0) > 0 else "—",
                        f"{1/analysis.odds.get('2', 99)*100:.1f}%" if analysis.odds.get('2', 0) > 0 else "—",
                    ],
                    "Cote": [
                        f"{analysis.odds.get('1', 0):.2f}",
                        f"{analysis.odds.get('X', 0):.2f}",
                        f"{analysis.odds.get('2', 0):.2f}",
                    ]
                }

                # Ajouter la colonne Value
                values = []
                for key in ["1", "X", "2"]:
                    mp = probs_1x2.get(key, 0)
                    bo = analysis.odds.get(key, 0)
                    if bo > 0 and mp > 0:
                        v = (bo * mp - 1) * 100
                        values.append(f"{v:+.1f}%")
                    else:
                        values.append("—")

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
                    data["Résultat"].append(label)
                    data["Modèle"].append(f"{mp*100:.1f}%")
                    data["Betclic"].append(f"{1/bo*100:.1f}%")
                    data["Cote"].append(f"{bo:.2f}")
                    if mp > 0:
                        v = (bo * mp - 1) * 100
                        data["Value"].append(f"{v:+.1f}%")
                    else:
                        data["Value"].append("—")

                st.dataframe(
                    pd.DataFrame(data),
                    use_container_width=True,
                    hide_index=True
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
                        line=dict(color='#00ff88', width=3),
                        fill='toself',
                        fillcolor='rgba(0, 255, 136, 0.1)'
                    ))

                    fig.add_trace(go.Scatterpolar(
                        r=betclic_vals + [betclic_vals[0]],
                        theta=categories + [categories[0]],
                        name='Betclic',
                        line=dict(color='#ff4444', width=3),
                        fill='toself',
                        fillcolor='rgba(255, 68, 68, 0.1)'
                    ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 70]),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        showlegend=True,
                        height=300,
                        margin=dict(l=40, r=40, t=20, b=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Value bets
            if analysis.value_bets:
                st.success(
                    f"🎯 **{analysis.total_value_bets} VALUE BET(S) DÉTECTÉ(S)**"
                )

                for vb in analysis.value_bets:
                    st.markdown(f"""
                    <div class="value-bet-card">
                        <strong>{vb.market}</strong> → <strong>{vb.selection}</strong>
                        @ <strong>{vb.bookmaker_odds:.2f}</strong><br>
                        📈 Value : <strong>+{vb.value_percentage*100:.1f}%</strong> {vb.value_rating} |
                        🎯 Confiance : {vb.confidence_score:.0f}/100 |
                        💰 Mise : <strong>{vb.recommended_stake:,.0f} FCFA</strong>
                        ({vb.kelly_stake:.1f}% bankroll)
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-value-card">
                    ❌ Aucun value bet détecté — les cotes semblent correctement calibrées
                </div>
                """, unsafe_allow_html=True)

            # Infos additionnelles
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.caption(f"🎯 Score prédit : **{analysis.predicted_score}**")
                st.caption(f"📈 Résultat probable : **{analysis.most_likely_result}**")
            with col_info2:
                st.caption(f"📐 Marge Betclic : **{analysis.bookmaker_margin:.1f}%**")
                st.caption(f"🔒 Confiance : **{analysis.analysis_confidence:.0f}/100**")

            if analysis.data_warning:
                st.warning(analysis.data_warning)

    # ── Tableau récapitulatif final ──
    all_vbs = []
    for a in analyses:
        for vb in a.value_bets:
            all_vbs.append({
                "Match": f"{a.home_team} vs {a.away_team}",
                "Marché": vb.market.split(" - ")[-1] if " - " in vb.market else vb.market,
                "Sélection": vb.selection,
                "Cote": vb.bookmaker_odds,
                "Value": f"+{vb.value_percentage*100:.1f}%",
                "Rating": vb.value_rating,
                "Confiance": f"{vb.confidence_score:.0f}",
                "Mise (FCFA)": f"{vb.recommended_stake:,.0f}",
            })

    if all_vbs:
        st.markdown("---")
        st.subheader("🏆 Tous les Value Bets Recommandés")
        st.dataframe(
            pd.DataFrame(all_vbs),
            use_container_width=True,
            hide_index=True
        )


# ══════════════════════════════════════════════════════
#  PAGE : TABLEAU DE BORD
# ══════════════════════════════════════════════════════

def page_dashboard():
    """Tableau de bord avec les statistiques globales."""

    st.markdown(
        '<div class="main-header">📊 Tableau de Bord</div>',
        unsafe_allow_html=True
    )

    db = init_modules()["db"]
    stats = db.get_performance_stats()

    if not stats or stats.get("total_bets", 0) == 0:
        st.info(
            "📌 Aucune donnée disponible. "
            "Commencez par analyser des matchs et enregistrer les résultats."
        )
        return

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎰 Total Paris",
            stats.get("total_bets", 0)
        )

    with col2:
        wr = stats.get("win_rate", 0)
        st.metric(
            "✅ Win Rate",
            f"{wr:.1f}%"
        )

    with col3:
        roi = stats.get("roi", 0)
        st.metric(
            "📈 ROI",
            f"{roi:+.1f}%",
            delta=f"{stats.get('total_profit', 0):+,.0f} FCFA"
        )

    with col4:
        st.metric(
            "🎯 Value Moyenne",
            f"{stats.get('avg_value', 0)*100:+.1f}%"
        )

    # Graphique d'évolution du bankroll
    backtester = init_modules()["backtester"]
    result = backtester.run_backtest()

    if result.bankroll_history:
        st.subheader("💰 Évolution du Bankroll")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=result.bankroll_history,
            mode='lines',
            name='Bankroll',
            line=dict(color='#00ff88', width=2),
            fill='toself',
            fillcolor='rgba(0, 255, 136, 0.05)'
        ))

        fig.update_layout(
            height=400,
            xaxis_title="Nombre de paris",
            yaxis_title="Bankroll (FCFA)",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=20, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
#  PAGE : BACKTESTING
# ══════════════════════════════════════════════════════

def page_backtesting():
    """Page de backtesting et validation."""

    st.markdown(
        '<div class="main-header">📈 Backtesting & Validation</div>',
        unsafe_allow_html=True
    )

    backtester = init_modules()["backtester"]

    # Charger les paris en attente
    db = init_modules()["db"]
    pending = db.get_pending_bets()

    if pending:
        st.subheader("⏳ Paris en attente de résultat")

        for bet in pending:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(
                    f"**{bet['home_team']} vs {bet['away_team']}** — "
                    f"{bet['market']} → {bet['selection']} @ {bet['bookmaker_odds']}"
                )

            with col2:
                if st.button("✅ Gagné", key=f"win_{bet['id']}"):
                    profit = bet['recommended_stake'] * (bet['bookmaker_odds'] - 1)
                    db.update_bet_result(bet['id'], "win", profit)
                    st.rerun()

            with col3:
                if st.button("❌ Perdu", key=f"loss_{bet['id']}"):
                    db.update_bet_result(bet['id'], "loss", -bet['recommended_stake'])
                    st.rerun()

    # Lancer le backtest
    st.divider()

    if st.button("🔄 Recalculer le Backtest", type="primary"):
        result = backtester.run_backtest()

        if result.total_bets > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 ROI", f"{result.roi:+.1f}%")
            col2.metric("📉 Max Drawdown", f"{result.max_drawdown:.1f}%")
            col3.metric(
                "📊 P-value",
                f"{result.p_value:.4f}",
                delta="Significatif" if result.is_significant else "Non significatif"
            )
        else:
            st.info("Aucun pari résolu disponible pour le backtest.")


# ══════════════════════════════════════════════════════
#  PAGE : PARAMÈTRES
# ══════════════════════════════════════════════════════

def page_settings():
    """Page de configuration des paramètres."""

    st.markdown(
        '<div class="main-header">⚙️ Paramètres</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎲 Modèle Poisson",
        "📊 Elo",
        "🎯 Value Bet",
        "🔑 APIs"
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
        st.subheader("🔑 Clés API")
        st.text_input("OpenAI API Key", type="password", value="sk-...")
        st.text_input("RapidAPI Key", type="password", value="")
        st.text_input("The Odds API Key", type="password", value="")

        if st.button("💾 Sauvegarder"):
            st.success("✅ Paramètres sauvegardés !")


# ══════════════════════════════════════════════════════
#  POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════

def main():
    """Point d'entrée de l'application Streamlit."""

    page, bankroll, min_value, min_confidence = render_sidebar()

    if "Analyse" in page or "🏠" in page:
        # Page d'accueil avec les deux options
        st.markdown(
            '<div class="main-header">🏠 Value Bet Analyzer — Betclic CI</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        ### 👋 Bienvenue dans le Value Bet Analyzer !

        Ce logiciel analyse automatiquement les cotes de **Betclic Côte d'Ivoire**
        pour identifier les **value bets** — des paris où la probabilité réelle
        est supérieure à ce que suggèrent les cotes.

        #### 🚀 Comment commencer ?

        1. **📸 Upload** — Envoyez vos captures d'écran Betclic
        2. **✍️ Saisie manuelle** — Entrez les matchs et cotes à la main
        3. **📊 Tableau de bord** — Suivez vos performances
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.info("📸 **Upload de captures d'écran**\n\nEnvoyez vos screenshots Betclic CI.")

        with col2:
            st.info("✍️ **Saisie manuelle**\n\nEntrez les matchs et cotes directement.")

    elif "Upload" in page or "📸" in page:
        page_upload_screenshots(bankroll, min_value, min_confidence)

    elif "Saisie" in page or "✍️" in page:
        page_manual_entry(bankroll, min_value, min_confidence)

    elif "Tableau" in page or "📊" in page:
        page_dashboard()

    elif "Backtesting" in page or "📈" in page:
        page_backtesting()

    elif "Paramètres" in page or "⚙️" in page:
        page_settings()


if __name__ == "__main__":
    main()
