"""
═══════════════════════════════════════════════════════
 MODULE SCORE FETCHER — Scores finaux automatiques
 via The Odds API (3 derniers jours)
═══════════════════════════════════════════════════════

L'endpoint /scores de The Odds API renvoie les scores des
matchs terminés des 3 derniers jours (2 crédits par ligue).
On y associe les matchs analysés en attente de résultat par
correspondance floue des noms d'équipes.

API-Football ne peut pas servir ici : son plan gratuit ne
couvre pas les saisons en cours.
"""

from typing import Dict, List, Optional, Tuple

import requests
from rapidfuzz import fuzz

from config import APIKeys
from modules.odds_collector import OddsAPICollector

FUZZY_MIN = 65


class ScoreFetcher:
    """Récupère les scores finaux récents via The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self):
        self.api_key = APIKeys.ODDS_API_KEY
        self.quota_restant = None

    def fetch_completed(self, days_from: int = 3) -> List[Dict]:
        """
        Scores des matchs TERMINÉS des `days_from` derniers jours,
        toutes ligues connues confondues (2 crédits par ligue).
        """

        if not self.api_key:
            return []

        events: List[Dict] = []

        for sport_key in set(OddsAPICollector.LEAGUE_KEYS.values()):
            try:
                r = requests.get(
                    f"{self.BASE_URL}/sports/{sport_key}/scores/",
                    params={
                        "apiKey": self.api_key,
                        "daysFrom": days_from,
                    },
                    timeout=20,
                )
                self.quota_restant = r.headers.get("x-requests-remaining")
                if r.status_code != 200:
                    continue

                for ev in r.json():
                    if not ev.get("completed"):
                        continue
                    scores = {s.get("name"): s.get("score")
                              for s in (ev.get("scores") or [])}
                    home, away = ev.get("home_team"), ev.get("away_team")
                    try:
                        fthg = int(scores.get(home))
                        ftag = int(scores.get(away))
                    except (TypeError, ValueError):
                        continue
                    events.append({
                        "home": home, "away": away,
                        "fthg": fthg, "ftag": ftag,
                    })
            except requests.RequestException:
                continue

        return events

    @staticmethod
    def find_score(home_team: str, away_team: str,
                   events: List[Dict]) -> Optional[Tuple[int, int]]:
        """
        Associe un match analysé à un score par correspondance
        floue des DEUX noms d'équipes (≥ 65 chacun).
        """

        best, best_score = None, 0
        for ev in events:
            s_home = fuzz.token_sort_ratio(
                home_team.lower(), ev["home"].lower())
            s_away = fuzz.token_sort_ratio(
                away_team.lower(), ev["away"].lower())
            if s_home >= FUZZY_MIN and s_away >= FUZZY_MIN:
                score = s_home + s_away
                if score > best_score:
                    best, best_score = ev, score

        if best:
            return best["fthg"], best["ftag"]
        return None
