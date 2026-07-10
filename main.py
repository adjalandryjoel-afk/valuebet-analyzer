"""
═══════════════════════════════════════════════════════════════════
    ⚽ VALUE BET ANALYZER v2.0 — BETCLIC CÔTE D'IVOIRE ⚽

    Version enrichie avec :
    • Scraping xG (FBref + Understat)
    • Modèle XGBoost Machine Learning
    • Cotes multi-bookmakers (The Odds API)
    • Conditions météo
    • Blessures & suspensions
    • Backtesting complet
    • Base de données SQLite
    • Interface Streamlit (via app.py)

    Usage :
        python main.py --demo        → Mode démonstration
        python main.py --bankroll N  → Bankroll personnalisé
        streamlit run app.py         → Interface graphique
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
from typing import List, Dict

from config import (
    Paths, KellyConfig, ValueBetConfig,
    PoissonConfig, OCRConfig, SUPPORTED_LEAGUES
)
from modules.ocr_extractor import BetclicScreenshotExtractor, create_extractor
from modules.team_matcher import TeamMatcher
from modules.data_collector import DataCollector, MatchContext
from modules.poisson_model import PoissonPredictor
from modules.elo_rating import EloRatingSystem
from modules.value_detector import ValueBetDetector, MatchAnalysis
from modules.kelly_criterion import KellyStakeCalculator, StakeRecommendation
from modules.report_generator import ReportGenerator
from modules.database_manager import DatabaseManager

# Nouveaux modules
from modules.xg_scraper import XGCollector
from modules.xgboost_model import XGBoostPredictor
from modules.weather_collector import WeatherCollector
from modules.injuries_scraper import InjuriesScraper
from modules.backtester import Backtester


def create_directories():
    """Crée les dossiers nécessaires."""
    for d in [Paths.SCREENSHOTS_DIR, Paths.REPORTS_DIR, Paths.DATA_DIR, "models"]:
        os.makedirs(d, exist_ok=True)


def run_demo_mode():
    """Mode démonstration avec données fictives."""

    print("\n🎮 MODE DÉMONSTRATION v2.0\n")

    return [
        {
            "equipe_domicile": "PSG",
            "equipe_exterieur": "Marseille",
            "competition": "Ligue 1 France",
            "cotes": {"1": 1.55, "X": 4.20, "2": 5.50},
            "marches_supplementaires": {
                "over_2_5": 1.72, "under_2_5": 2.10,
                "btts_oui": 1.85, "btts_non": 1.95
            }
        },
        {
            "equipe_domicile": "ASEC Mimosas",
            "equipe_exterieur": "Africa Sports",
            "competition": "Ligue 1 Côte d'Ivoire",
            "cotes": {"1": 1.85, "X": 3.40, "2": 4.20},
            "marches_supplementaires": {
                "over_2_5": 2.25, "under_2_5": 1.62,
                "btts_oui": 2.05, "btts_non": 1.75
            }
        },
        {
            "equipe_domicile": "Real Madrid",
            "equipe_exterieur": "Barcelona",
            "competition": "La Liga",
            "cotes": {"1": 2.20, "X": 3.50, "2": 3.10},
            "marches_supplementaires": {
                "over_2_5": 1.65, "under_2_5": 2.20,
                "btts_oui": 1.60, "btts_non": 2.30
            }
        },
        {
            "equipe_domicile": "Manchester City",
            "equipe_exterieur": "Liverpool",
            "competition": "Premier League",
            "cotes": {"1": 1.90, "X": 3.80, "2": 3.60},
            "marches_supplementaires": {
                "over_2_5": 1.55, "under_2_5": 2.40,
                "btts_oui": 1.55, "btts_non": 2.40
            }
        },
        {
            "equipe_domicile": "Arsenal",
            "equipe_exterieur": "Chelsea",
            "competition": "Premier League",
            "cotes": {"1": 1.75, "X": 3.90, "2": 4.50},
            "marches_supplementaires": {
                "over_2_5": 1.80, "under_2_5": 2.00,
                "btts_oui": 1.75, "btts_non": 2.05
            }
        }
    ]


def process_matches_v2(matches: List[Dict], bankroll: float,
                        enable_xg: bool = True,
                        enable_xgboost: bool = False,
                        enable_weather: bool = False,
                        enable_injuries: bool = False):
    """
    Pipeline v2 : traite tous les matchs avec les modules avancés.
    """

    # ── Initialiser TOUS les modules ──
    team_matcher = TeamMatcher()
    data_collector = DataCollector()
    poisson = PoissonPredictor()
    elo = EloRatingSystem()
    value_detector = ValueBetDetector()
    kelly = KellyStakeCalculator(bankroll=bankroll)
    report = ReportGenerator(bankroll=bankroll)
    db = DatabaseManager()
    backtester = Backtester(starting_bankroll=bankroll)

    # Modules avancés (optionnels)
    xg_collector = XGCollector() if enable_xg else None
    xgboost = XGBoostPredictor() if enable_xgboost else None
    weather = WeatherCollector() if enable_weather else None
    injuries = InjuriesScraper() if enable_injuries else None

    report.print_header()

    all_analyses: List[MatchAnalysis] = []
    all_stakes: List[StakeRecommendation] = []

    print(f"\n{'━'*70}")
    print(f"  🔄 TRAITEMENT DE {len(matches)} MATCH(S) — V2.0 AVANCÉ")
    print(f"{'━'*70}")

    if enable_xg:
        print(f"  📊 Module xG : ✅ Activé (FBref + Understat)")
    if enable_xgboost:
        print(f"  🤖 Module XGBoost : ✅ Activé")
    if enable_weather:
        print(f"  🌤️ Module Météo : ✅ Activé")
    if enable_injuries:
        print(f"  🏥 Module Blessures : ✅ Activé")

    for i, match in enumerate(matches, 1):
        home_raw = match.get("equipe_domicile", "")
        away_raw = match.get("equipe_exterieur", "")
        cotes = match.get("cotes", {})
        extras = match.get("marches_supplementaires", {})
        competition = match.get("competition", "")

        print(f"\n  ┌─ Match {i}/{len(matches)} {'─'*40}")
        print(f"  │ {home_raw} vs {away_raw}")

        # ── ÉTAPE 1 : Identification ──
        match_info = team_matcher.identify_match(home_raw, away_raw)
        home_name = match_info["home"]["official_name"]
        away_name = match_info["away"]["official_name"]
        league = match_info.get("league", "unknown")

        print(f"  │ 🔍 {home_name} vs {away_name} ({league})")

        # ── ÉTAPE 2 : Collecte des données de base ──
        all_odds = {**cotes, **extras}
        context = data_collector.collect_match_data(match_info, all_odds)
        context.competition = competition

        # ── ÉTAPE 3 : Enrichissement xG ──
        home_xg_profile = None
        away_xg_profile = None

        if xg_collector and league in ["premier_league", "la_liga", "serie_a",
                                        "bundesliga", "ligue1_fr"]:
            print(f"  │ 📊 Chargement des xG...")
            try:
                context, home_xg_profile, away_xg_profile = (
                    xg_collector.enrich_match_context(
                        context, home_name, away_name, league
                    )
                )
            except Exception as e:
                print(f"  │ ⚠️ xG indisponible : {e}")

        # ── ÉTAPE 4 : Météo ──
        if weather and match_info["home"].get("info"):
            city = match_info["home"]["info"].get("city", "")
            if city:
                print(f"  │ 🌤️ Météo {city}...")
                try:
                    match_weather = weather.get_weather(city)
                    if match_weather:
                        print(f"  │    {match_weather.temperature}°C, "
                              f"{match_weather.description} "
                              f"({match_weather.impact_description})")
                except Exception:
                    pass

        # ── ÉTAPE 5 : Blessures ──
        if injuries:
            print(f"  │ 🏥 Vérification des blessures...")
            try:
                home_abs = injuries.get_team_absences(home_name)
                away_abs = injuries.get_team_absences(away_name)

                if home_abs and home_abs.total_absent > 0:
                    impact = injuries.calculate_absence_impact(home_abs)
                    print(f"  │    {home_name}: {impact['description']}")

                if away_abs and away_abs.total_absent > 0:
                    impact = injuries.calculate_absence_impact(away_abs)
                    print(f"  │    {away_name}: {impact['description']}")
            except Exception:
                pass

        # ── ÉTAPE 6 : Elo ──
        if cotes.get("1", 0) > 0 and cotes.get("2", 0) > 0:
            elo.estimate_rating_from_odds(
                home_name, cotes["1"], cotes["2"], is_home=True
            )
            elo.estimate_rating_from_odds(
                away_name, cotes["2"], cotes["1"], is_home=False
            )

        elo_pred = elo.predict(home_name, away_name)
        print(f"  │ 📈 Elo: {home_name}={elo_pred.home_rating:.0f} "
              f"| {away_name}={elo_pred.away_rating:.0f}")

        # ── ÉTAPE 7 : Poisson ──
        poisson_pred = poisson.predict(context)
        print(f"  │ 🎲 Poisson: λ_dom={poisson_pred.lambda_home:.2f} "
              f"λ_ext={poisson_pred.lambda_away:.2f}")

        # ── ÉTAPE 8 : XGBoost (si entraîné) ──
        xgb_pred = None
        if xgboost and xgboost.is_trained:
            print(f"  │ 🤖 XGBoost...")
            xgb_pred = xgboost.predict(
                context,
                home_elo=elo_pred.home_rating,
                away_elo=elo_pred.away_rating,
                home_xg_profile=home_xg_profile,
                away_xg_profile=away_xg_profile
            )
            print(f"  │    H:{xgb_pred.prob_home_win:.1%} "
                  f"D:{xgb_pred.prob_draw:.1%} "
                  f"A:{xgb_pred.prob_away_win:.1%}")

        # ── ÉTAPE 9 : Détection value bets ──
        analysis = value_detector.analyze_match(
            home_name, away_name, all_odds,
            poisson_pred, elo_pred, competition
        )

        # ── ÉTAPE 10 : Mises Kelly ──
        match_stakes = kelly.calculate_all_stakes(analysis)

        if analysis.has_value:
            print(f"  │ ✅ {analysis.total_value_bets} value bet(s) !")
        else:
            print(f"  │ ❌ Pas de value bet")

        print(f"  └{'─'*50}")

        # ── SAUVEGARDER EN BASE ──
        try:
            match_id = db.save_match_analysis({
                "home_team": home_name,
                "away_team": away_name,
                "competition": competition,
                "match_date": match.get("date"),
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
                "lambda_home": poisson_pred.lambda_home,
                "lambda_away": poisson_pred.lambda_away,
                "elo_home": elo_pred.home_rating,
                "elo_away": elo_pred.away_rating,
                "xg_home": home_xg_profile.avg_xg_for if home_xg_profile else None,
                "xg_away": away_xg_profile.avg_xg_for if away_xg_profile else None,
                "predicted_score": analysis.predicted_score,
                "predicted_result": analysis.most_likely_result,
                "confidence": analysis.analysis_confidence,
                "bookmaker_margin": analysis.bookmaker_margin,
            })

            for vb in analysis.value_bets:
                db.save_value_bet(match_id, {
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
        except Exception as e:
            print(f"  ⚠️ Erreur sauvegarde DB : {e}")

        all_analyses.append(analysis)
        all_stakes.extend(match_stakes)

    # ── Ajustement portfolio ──
    if all_stakes:
        all_stakes = kelly.portfolio_adjustment(all_stakes)

    # ── Rapports ──
    print(f"\n\n{'▓'*70}")
    print(f"  📋 RAPPORTS DÉTAILLÉS PAR MATCH")
    print(f"{'▓'*70}")

    for analysis in all_analyses:
        match_stakes = [
            s for s in all_stakes
            if s.value_bet.match == f"{analysis.home_team} vs {analysis.away_team}"
        ]
        report.print_match_report(analysis, match_stakes)

    report.print_summary(all_analyses, all_stakes)
    report.export_to_json(all_analyses)
    elo.save_ratings()

    return all_analyses, all_stakes


def main():
    """Point d'entrée principal v2."""

    parser = argparse.ArgumentParser(
        description="⚽ Value Bet Analyzer v2.0 — Betclic Côte d'Ivoire"
    )
    parser.add_argument("--bankroll", type=float,
                        default=KellyConfig.DEFAULT_BANKROLL)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--screenshots-dir", type=str,
                        default=OCRConfig.SCREENSHOTS_DIR)
    parser.add_argument("--enable-xg", action="store_true", default=True,
                        help="Activer le scraping xG")
    parser.add_argument("--enable-xgboost", action="store_true", default=False,
                        help="Activer le modèle XGBoost")
    parser.add_argument("--enable-weather", action="store_true", default=False,
                        help="Activer la météo")
    parser.add_argument("--enable-injuries", action="store_true", default=False,
                        help="Activer le scraping des blessures")

    args = parser.parse_args()

    create_directories()

    if args.demo:
        matches = run_demo_mode()
    else:
        extractor = create_extractor()
        matches = extractor.extract_from_directory(args.screenshots_dir)

        if not matches:
            print("\n❌ Aucun match trouvé.")
            print("💡 Lancez : python main.py --demo")
            sys.exit(1)

    analyses, stakes = process_matches_v2(
        matches, args.bankroll,
        enable_xg=args.enable_xg,
        enable_xgboost=args.enable_xgboost,
        enable_weather=args.enable_weather,
        enable_injuries=args.enable_injuries
    )

    total_vb = sum(a.total_value_bets for a in analyses)
    print(f"\n🏁 Analyse v2.0 terminée ! {total_vb} value bet(s) sur {len(analyses)} match(s).\n")


if __name__ == "__main__":
    main()
