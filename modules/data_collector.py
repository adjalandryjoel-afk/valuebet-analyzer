"""
═══════════════════════════════════════════════════════
 MODULE DATA COLLECTOR — Construction du contexte
 statistique d'un match
═══════════════════════════════════════════════════════

Assemble tout ce que le modèle sait sur un match :
stats des équipes, moyenne de buts de la ligue, cotes.
Sans API, les stats de base sont estimées depuis les
probabilités implicites du marché (les cotes contiennent
déjà énormément d'information).
"""

import os
import json
import math
from typing import Dict, Optional
from dataclasses import dataclass, field

from config import Paths, PoissonConfig, SUPPORTED_LEAGUES
from modules.api_football import ApiFootballCollector


# ══════════════════════════════════════════════════════
#  STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════

@dataclass
class TeamStats:
    """Statistiques d'une équipe utilisées par les modèles."""

    team_name: str = ""

    # Buts (moyennes par match)
    avg_goals_scored: float = 1.30
    avg_goals_conceded: float = 1.30
    avg_goals_scored_home: float = 1.45
    avg_goals_conceded_home: float = 1.15
    avg_goals_scored_away: float = 1.10
    avg_goals_conceded_away: float = 1.40

    # xG (remplis par le module xg_scraper si dispo)
    xg_scored: float = 0.0
    xg_conceded: float = 0.0
    xg_available: bool = False

    # Forme récente (points par match sur les 5 derniers, 0..3)
    recent_form_score: float = 1.50

    # Classement
    league_position: int = 10
    points_per_game: float = 1.50

    matches_played: int = 0
    data_source: str = "estimated"  # "estimated", "historical", "api"


@dataclass
class MatchContext:
    """Contexte complet d'un match pour les modèles de prédiction."""

    home_team: str = ""
    away_team: str = ""
    competition: str = ""
    league: str = "unknown"

    home_stats: Optional[TeamStats] = None
    away_stats: Optional[TeamStats] = None

    # Cotes Betclic (clés : "1", "X", "2", "over_2_5", ...)
    odds: Dict = field(default_factory=dict)

    # Moyenne de buts de la ligue
    league_avg_goals: float = PoissonConfig.DEFAULT_LEAGUE_AVG_GOALS

    # Qualité des données disponibles (0-100)
    data_completeness: float = 0.0


# ══════════════════════════════════════════════════════
#  COLLECTEUR
# ══════════════════════════════════════════════════════

class DataCollector:
    """
    Construit le MatchContext d'un match.

    Sources par ordre de priorité :
    1. API-Football (stats réelles, indépendantes des cotes)
    2. data/historical_data.json (stats sauvegardées)
    3. Estimation depuis les cotes (toujours disponible)
    """

    def __init__(self):
        self.historical = self._load_historical()
        self.api_collector = ApiFootballCollector()

    def _load_historical(self) -> Dict:
        """Charge les stats historiques sauvegardées si présentes."""

        if os.path.exists(Paths.HISTORICAL_DATA):
            try:
                with open(Paths.HISTORICAL_DATA, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    # ─── CONSTRUCTION DU CONTEXTE ───────────────────

    def collect_match_data(self, match_info: Dict, odds: Dict) -> MatchContext:
        """
        Construit le MatchContext depuis le résultat du TeamMatcher
        et les cotes extraites.
        """

        home_name = match_info["home"]["official_name"]
        away_name = match_info["away"]["official_name"]
        league = match_info.get("league", "unknown")

        league_info = SUPPORTED_LEAGUES.get(league, {})
        league_avg = league_info.get(
            "avg_goals", PoissonConfig.DEFAULT_LEAGUE_AVG_GOALS
        )

        context = MatchContext(
            home_team=home_name,
            away_team=away_name,
            league=league,
            odds=dict(odds or {}),
            league_avg_goals=league_avg,
        )

        # Stats des équipes :
        # API-Football > historique > estimation par les cotes
        context.home_stats = (
            self._stats_from_api(home_name)
            or self._stats_from_historical(home_name)
            or self._estimate_stats_from_odds(home_name, odds, league_avg,
                                              is_home=True)
        )
        context.away_stats = (
            self._stats_from_api(away_name)
            or self._stats_from_historical(away_name)
            or self._estimate_stats_from_odds(away_name, odds, league_avg,
                                              is_home=False)
        )

        # Score de complétude
        completeness = 30.0  # cotes 1X2 = base
        if odds.get("over_2_5") and odds.get("under_2_5"):
            completeness += 15
        if odds.get("btts_oui") and odds.get("btts_non"):
            completeness += 10
        if context.home_stats.data_source == "api":
            completeness += 20
        elif context.home_stats.data_source == "historical":
            completeness += 15
        if context.away_stats.data_source == "api":
            completeness += 20
        elif context.away_stats.data_source == "historical":
            completeness += 15
        if match_info.get("both_matched"):
            completeness += 10

        context.data_completeness = min(completeness, 100.0)

        return context

    # ─── STATS DEPUIS API-FOOTBALL ──────────────────

    def _stats_from_api(self, team_name: str) -> Optional[TeamStats]:
        """
        Récupère les stats réelles d'une équipe via API-Football.

        Ce sont les seules stats totalement indépendantes des cotes :
        indispensables pour détecter de la value sur Over/Under et BTTS.
        """

        api_stats = self.api_collector.get_team_stats(team_name)
        if not api_stats:
            return None

        stats = TeamStats(team_name=team_name, data_source="api")

        for attr in (
            "avg_goals_scored", "avg_goals_conceded",
            "avg_goals_scored_home", "avg_goals_conceded_home",
            "avg_goals_scored_away", "avg_goals_conceded_away",
            "recent_form_score", "matches_played",
        ):
            if attr in api_stats:
                setattr(stats, attr, api_stats[attr])

        # Non fournis par l'API : points_per_game approximé depuis
        # la forme récente, league_position garde son défaut
        stats.points_per_game = stats.recent_form_score

        return stats

    # ─── STATS DEPUIS L'HISTORIQUE ──────────────────

    def _stats_from_historical(self, team_name: str) -> Optional[TeamStats]:
        """Récupère les stats sauvegardées d'une équipe si disponibles."""

        record = self.historical.get(team_name)
        if not record:
            return None

        stats = TeamStats(team_name=team_name, data_source="historical")

        for attr in (
            "avg_goals_scored", "avg_goals_conceded",
            "avg_goals_scored_home", "avg_goals_conceded_home",
            "avg_goals_scored_away", "avg_goals_conceded_away",
            "recent_form_score", "league_position",
            "points_per_game", "matches_played",
        ):
            if attr in record:
                setattr(stats, attr, record[attr])

        return stats

    # ─── ESTIMATION DEPUIS LES COTES ────────────────

    def _estimate_stats_from_odds(self, team_name: str, odds: Dict,
                                   league_avg: float,
                                   is_home: bool) -> TeamStats:
        """
        Estime des stats plausibles depuis les probabilités
        implicites du marché.

        Le marché est efficient : une équipe cotée à 1.55 est
        objectivement bien plus forte que son adversaire. On
        traduit cet écart en buts attendus.
        """

        stats = TeamStats(team_name=team_name, data_source="estimated")

        o1 = float(odds.get("1", 0) or 0)
        ox = float(odds.get("X", 0) or 0)
        o2 = float(odds.get("2", 0) or 0)

        if o1 <= 1 or o2 <= 1:
            return stats  # pas de cotes exploitables → défauts

        # Probabilités no-vig
        inv = [1 / o1, (1 / ox) if ox > 1 else 0.25, 1 / o2]
        total = sum(inv)
        p_home, p_draw, p_away = (v / total for v in inv)

        # Total de buts attendu : depuis over/under 2.5 si dispo
        total_goals = league_avg
        o_over = float(odds.get("over_2_5", 0) or 0)
        o_under = float(odds.get("under_2_5", 0) or 0)
        if o_over > 1 and o_under > 1:
            p_over = (1 / o_over) / (1 / o_over + 1 / o_under)
            total_goals = self._total_goals_from_over25(p_over)

        # Répartition du total selon la force relative
        p_team = p_home if is_home else p_away
        p_opp = p_away if is_home else p_home

        # part des buts ∝ force relative, bornée pour rester réaliste
        share = 0.5 + 0.55 * (p_team - p_opp)
        share = max(0.25, min(0.75, share))

        expected_for = total_goals * share
        expected_against = total_goals * (1 - share)

        stats.avg_goals_scored = round(expected_for, 2)
        stats.avg_goals_conceded = round(expected_against, 2)

        # Déclinaisons domicile/extérieur (facteurs standards)
        stats.avg_goals_scored_home = round(expected_for * 1.12, 2)
        stats.avg_goals_conceded_home = round(expected_against * 0.90, 2)
        stats.avg_goals_scored_away = round(expected_for * 0.88, 2)
        stats.avg_goals_conceded_away = round(expected_against * 1.10, 2)

        # Forme et classement approximés depuis la force
        stats.recent_form_score = round(0.5 + 2.5 * p_team, 2)
        stats.points_per_game = round(0.4 + 2.2 * p_team, 2)
        stats.league_position = max(1, min(20, round(20 - 19 * p_team)))

        return stats

    @staticmethod
    def _total_goals_from_over25(p_over: float) -> float:
        """
        Inverse P(N ≥ 3) = p_over pour N ~ Poisson(λ) et retourne λ.

        Recherche dichotomique — précis et sans dépendance scipy.
        """

        p_over = max(0.02, min(0.98, p_over))

        def prob_over(lam: float) -> float:
            # P(N >= 3) = 1 - P(0) - P(1) - P(2)
            p0 = math.exp(-lam)
            p1 = p0 * lam
            p2 = p1 * lam / 2
            return 1 - (p0 + p1 + p2)

        lo, hi = 0.2, 6.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if prob_over(mid) < p_over:
                lo = mid
            else:
                hi = mid

        return round((lo + hi) / 2, 3)
