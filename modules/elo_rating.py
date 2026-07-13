"""
═══════════════════════════════════════════════════════
 MODULE ELO — Système de rating Elo pour le football
═══════════════════════════════════════════════════════

Chaque équipe a un rating (1500 = moyen). La différence
de ratings donne une espérance de victoire. Les ratings
sont persistés dans data/elo_ratings.json et peuvent être
initialisés depuis les cotes du marché quand une équipe
est inconnue.

Source prioritaire : ClubElo (clubelo.com), un vrai Elo
européen indépendant des cotes — évite la circularité avec
l'ancrage marché. Repli : ratings estimés depuis les cotes.
ATTENTION : les deux échelles sont incompatibles (~1900 vs
~1550 pour un top club) et ne sont jamais mélangées.
"""

import os
import json
import math
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

from config import Paths, EloConfig
from modules.odds_utils import novig_probs

try:
    from modules.clubelo import ClubEloProvider
except ImportError:
    ClubEloProvider = None


@dataclass
class EloPrediction:
    """Prédiction issue du modèle Elo."""

    home_team: str = ""
    away_team: str = ""
    home_rating: float = EloConfig.INITIAL_RATING
    away_rating: float = EloConfig.INITIAL_RATING

    # Espérance de victoire domicile (avec avantage terrain, hors nul)
    home_expectancy: float = 0.5

    # Probabilités 1X2 (avec modèle de nul)
    prob_home_win: float = 0.0
    prob_draw: float = 0.0
    prob_away_win: float = 0.0

    model_name: str = "Elo"

    # Provenance des ratings : "clubelo" (Elo réel, les deux
    # équipes trouvées) ou "estimé" (repli depuis les cotes)
    elo_source: str = "estimé"


class EloRatingSystem:
    """Gestion des ratings Elo des équipes."""

    def __init__(self, clubelo=None):
        self.ratings: Dict[str, float] = {}
        self._load_ratings()

        # Fournisseur ClubElo (Elo réel indépendant des cotes)
        if clubelo is not None:
            self.clubelo = clubelo
        elif ClubEloProvider is not None:
            try:
                self.clubelo = ClubEloProvider()
            except Exception:
                self.clubelo = None
        else:
            self.clubelo = None

    # ─── PERSISTENCE ────────────────────────────────

    def _load_ratings(self):
        """Charge les ratings depuis data/elo_ratings.json."""

        if os.path.exists(Paths.ELO_RATINGS):
            try:
                with open(Paths.ELO_RATINGS, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.ratings = data.get("ratings", {})
            except (json.JSONDecodeError, OSError):
                self.ratings = {}

    def save_ratings(self):
        """Sauvegarde les ratings."""

        os.makedirs(Paths.DATA_DIR, exist_ok=True)

        with open(Paths.ELO_RATINGS, 'w', encoding='utf-8') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "ratings": self.ratings,
            }, f, ensure_ascii=False, indent=2)

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, EloConfig.INITIAL_RATING)

    # ─── ESTIMATION DEPUIS LES COTES ────────────────

    def estimate_rating_from_odds(self, team: str, odds_for: float,
                                   odds_against: float,
                                   is_home: bool = True) -> float:
        """
        Estime le rating d'une équipe depuis les cotes du marché.

        p(victoire) no-vig → différence Elo implicite → rating
        centré sur 1500, corrigé de l'avantage domicile.
        Si l'équipe a déjà un rating, on fait une moyenne pondérée.
        """

        if odds_for <= 1 or odds_against <= 1:
            return self.get_rating(team)

        # Probabilité de victoire relative sans le nul (Shin :
        # corrige le biais favori-outsider)
        p = novig_probs([odds_for, odds_against])[0]
        p = max(0.03, min(0.97, p))

        # Différence Elo implicite : p = 1 / (1 + 10^(-diff/400))
        diff = -400 * math.log10(1 / p - 1)

        # Retirer l'avantage domicile de l'estimation
        if is_home:
            diff -= EloConfig.HOME_ADVANTAGE_ELO
        else:
            diff += EloConfig.HOME_ADVANTAGE_ELO

        # La moitié de l'écart est attribuée à cette équipe
        estimated = EloConfig.INITIAL_RATING + diff / 2

        # Fusion avec le rating existant
        if team in self.ratings:
            w = EloConfig.ODDS_ESTIMATE_WEIGHT
            new_rating = w * estimated + (1 - w) * self.ratings[team]
        else:
            new_rating = estimated

        self.ratings[team] = round(new_rating, 1)
        return self.ratings[team]

    # ─── PRÉDICTION ─────────────────────────────────

    def predict(self, home_team: str, away_team: str) -> EloPrediction:
        """
        Prédit les probabilités 1X2 depuis les ratings.

        Modèle de nul : la probabilité de nul est maximale quand
        les équipes sont proches et diminue avec l'écart de niveau.

        Ratings : ClubElo si les DEUX équipes y sont trouvées
        (l'échelle ClubElo est directement compatible avec la
        logistique diff/400 et l'avantage domicile ~60 pts),
        sinon repli sur les ratings estimés depuis les cotes.
        Les deux échelles ne sont JAMAIS mélangées.
        """

        elo_source = "estimé"
        home_rating = away_rating = None

        if self.clubelo is not None:
            ce_home = self.clubelo.get_rating(home_team)
            ce_away = self.clubelo.get_rating(away_team)
            if ce_home is not None and ce_away is not None:
                home_rating, away_rating = ce_home, ce_away
                elo_source = "clubelo"

        if home_rating is None or away_rating is None:
            home_rating = self.get_rating(home_team)
            away_rating = self.get_rating(away_team)

        # Espérance avec avantage domicile
        diff = (home_rating + EloConfig.HOME_ADVANTAGE_ELO) - away_rating
        expectancy = 1 / (1 + 10 ** (-diff / 400))

        # Probabilité de nul (décroît avec |diff|)
        prob_draw = EloConfig.DRAW_BASE_PROB * math.exp(
            -abs(diff) / 600
        )
        prob_draw = max(0.08, min(0.35, prob_draw))

        # Répartir le reste selon l'espérance
        remaining = 1 - prob_draw
        prob_home = remaining * expectancy
        prob_away = remaining * (1 - expectancy)

        return EloPrediction(
            home_team=home_team,
            away_team=away_team,
            home_rating=home_rating,
            away_rating=away_rating,
            home_expectancy=round(expectancy, 4),
            prob_home_win=round(prob_home, 4),
            prob_draw=round(prob_draw, 4),
            prob_away_win=round(prob_away, 4),
            elo_source=elo_source,
        )

    # ─── MISE À JOUR APRÈS MATCH ────────────────────

    def record_result(self, home_team: str, away_team: str,
                      home_goals: int, away_goals: int):
        """
        Met à jour les ratings après un résultat réel.

        Le facteur K est modulé par l'écart de buts (une victoire
        3-0 fait bouger les ratings plus qu'un 1-0).
        """

        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        diff = (home_rating + EloConfig.HOME_ADVANTAGE_ELO) - away_rating
        expected_home = 1 / (1 + 10 ** (-diff / 400))

        if home_goals > away_goals:
            actual = 1.0
        elif home_goals == away_goals:
            actual = 0.5
        else:
            actual = 0.0

        # Modulation par l'écart de buts
        goal_diff = abs(home_goals - away_goals)
        multiplier = 1.0 if goal_diff <= 1 else 1 + 0.5 * math.log(goal_diff)

        delta = EloConfig.K_FACTOR * multiplier * (actual - expected_home)

        self.ratings[home_team] = round(home_rating + delta, 1)
        self.ratings[away_team] = round(away_rating - delta, 1)
