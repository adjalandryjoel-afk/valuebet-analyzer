"""
═══════════════════════════════════════════════════════
 MODULE ODDS COLLECTOR — Collecte de cotes en temps
 réel depuis plusieurs bookmakers via The Odds API
═══════════════════════════════════════════════════════

Permet de :
  • Comparer les cotes Betclic à d'autres bookmakers
  • Calculer les probabilités consensus du marché
  • Détecter les mouvements de cotes (steam moves)
  • Identifier le "sharp" money
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import APIKeys


@dataclass
class OddsSnapshot:
    """Snapshot des cotes d'un bookmaker à un instant T."""

    bookmaker: str
    odds_1: float = 0.0
    odds_x: float = 0.0
    odds_2: float = 0.0
    odds_over25: float = 0.0
    odds_under25: float = 0.0
    timestamp: str = ""
    margin: float = 0.0


@dataclass
class MarketConsensus:
    """Consensus du marché sur un match."""

    home_team: str
    away_team: str

    # Nombre de bookmakers
    num_bookmakers: int = 0

    # Probabilités consensus (moyenne sans marge)
    consensus_prob_1: float = 0.0
    consensus_prob_x: float = 0.0
    consensus_prob_2: float = 0.0

    # Cotes moyennes
    avg_odds_1: float = 0.0
    avg_odds_x: float = 0.0
    avg_odds_2: float = 0.0

    # Meilleures cotes
    best_odds_1: float = 0.0
    best_odds_x: float = 0.0
    best_odds_2: float = 0.0
    best_bookmaker_1: str = ""
    best_bookmaker_x: str = ""
    best_bookmaker_2: str = ""

    # Pinnacle (benchmark)
    pinnacle_odds_1: float = 0.0
    pinnacle_odds_x: float = 0.0
    pinnacle_odds_2: float = 0.0
    pinnacle_prob_1: float = 0.0
    pinnacle_prob_x: float = 0.0
    pinnacle_prob_2: float = 0.0

    # Cotes Betclic pour comparaison
    betclic_odds_1: float = 0.0
    betclic_odds_x: float = 0.0
    betclic_odds_2: float = 0.0

    # Marge moyenne du marché
    avg_margin: float = 0.0

    # Tous les snapshots
    snapshots: List[OddsSnapshot] = field(default_factory=list)


class OddsAPICollector:
    """
    Collecte les cotes de plusieurs bookmakers via The Odds API.

    The Odds API agrège les cotes de 40+ bookmakers mondiaux
    en temps réel.

    Plan gratuit : 500 requêtes/mois
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    # Sports IDs
    SPORT_FOOTBALL = "soccer"

    # Ligues disponibles
    LEAGUE_KEYS = {
        "premier_league": "soccer_epl",
        "la_liga": "soccer_spain_la_liga",
        "serie_a": "soccer_italy_serie_a",
        "bundesliga": "soccer_germany_bundesliga",
        "ligue1_fr": "soccer_france_ligue_one",
        "champions_league": "soccer_uefa_champs_league",
        "europa_league": "soccer_uefa_europa_league",
    }

    # Bookmakers africains / francophones à surveiller
    TARGET_BOOKMAKERS = [
        "betclic",
        "pinnacle",
        "bet365",
        "unibet",
        "williamhill",
        "1xbet",
        "betway",
        "marathonbet",
        "bwin",
    ]

    def __init__(self):
        self.api_key = APIKeys.ODDS_API_KEY
        self._remaining_requests = None

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Effectue une requête à The Odds API."""

        if params is None:
            params = {}

        params["apiKey"] = self.api_key

        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.get(url, params=params, timeout=30)

            # Suivre le quota
            self._remaining_requests = response.headers.get(
                'x-requests-remaining', '?'
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("      ❌ Clé API Odds invalide")
                return None
            elif response.status_code == 429:
                print("      ⏳ Quota Odds API épuisé")
                return None
            else:
                print(f"      ❌ Erreur Odds API: {response.status_code}")
                return None

        except Exception as e:
            print(f"      ❌ Erreur connexion Odds API: {e}")
            return None

    # ─── RÉCUPÉRER LES COTES D'UN MATCH ─────────────

    def get_match_odds(self, league: str, home_team: str = None,
                       away_team: str = None) -> List[MarketConsensus]:
        """
        Récupère les cotes de tous les bookmakers pour les matchs
        d'une ligue donnée.

        Args:
            league: Clé de la ligue (ex: "premier_league")
            home_team: Filtrer par équipe domicile (optionnel)
            away_team: Filtrer par équipe extérieur (optionnel)
        """

        sport_key = self.LEAGUE_KEYS.get(league)
        if not sport_key:
            print(f"      ⚠️ Ligue '{league}' non supportée par The Odds API")
            return []

        # Requête principale
        data = self._make_request(f"sports/{sport_key}/odds", {
            "regions": "eu,uk",
            "markets": "h2h,totals",
            "oddsFormat": "decimal"
        })

        if not data:
            return []

        results = []

        for event in data:
            event_home = event.get("home_team", "")
            event_away = event.get("away_team", "")

            # Filtrer si un nom d'équipe est spécifié
            if home_team:
                from rapidfuzz import fuzz
                if (fuzz.token_sort_ratio(home_team.lower(), event_home.lower()) < 60 and
                    fuzz.token_sort_ratio(away_team.lower() if away_team else "",
                                          event_away.lower()) < 60):
                    continue

            consensus = self._build_consensus(event)
            results.append(consensus)

        print(f"      ✅ {len(results)} match(s) avec cotes multi-bookmakers")
        print(f"      📊 Requêtes restantes : {self._remaining_requests}")

        return results

    def _build_consensus(self, event: Dict) -> MarketConsensus:
        """Construit le consensus du marché à partir des données de l'API."""

        consensus = MarketConsensus(
            home_team=event.get("home_team", ""),
            away_team=event.get("away_team", "")
        )

        all_odds_1 = []
        all_odds_x = []
        all_odds_2 = []

        for bookmaker in event.get("bookmakers", []):
            bk_name = bookmaker.get("key", "")

            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = market.get("outcomes", [])

                    snapshot = OddsSnapshot(
                        bookmaker=bk_name,
                        timestamp=bookmaker.get("last_update", "")
                    )

                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        price = float(outcome.get("price", 0))

                        if name == event.get("home_team"):
                            snapshot.odds_1 = price
                            all_odds_1.append(price)
                        elif name == "Draw":
                            snapshot.odds_x = price
                            all_odds_x.append(price)
                        else:
                            snapshot.odds_2 = price
                            all_odds_2.append(price)

                    # Calculer la marge
                    if snapshot.odds_1 > 0 and snapshot.odds_x > 0 and snapshot.odds_2 > 0:
                        snapshot.margin = (
                            1/snapshot.odds_1 +
                            1/snapshot.odds_x +
                            1/snapshot.odds_2 - 1
                        ) * 100

                    consensus.snapshots.append(snapshot)

                    # Pinnacle
                    if bk_name == "pinnacle":
                        consensus.pinnacle_odds_1 = snapshot.odds_1
                        consensus.pinnacle_odds_x = snapshot.odds_x
                        consensus.pinnacle_odds_2 = snapshot.odds_2

                    # Betclic
                    if "betclic" in bk_name.lower():
                        consensus.betclic_odds_1 = snapshot.odds_1
                        consensus.betclic_odds_x = snapshot.odds_x
                        consensus.betclic_odds_2 = snapshot.odds_2

        # Calculer les moyennes et meilleures cotes
        consensus.num_bookmakers = len(consensus.snapshots)

        if all_odds_1:
            consensus.avg_odds_1 = sum(all_odds_1) / len(all_odds_1)
            consensus.best_odds_1 = max(all_odds_1)
            best_idx = all_odds_1.index(max(all_odds_1))
            consensus.best_bookmaker_1 = consensus.snapshots[best_idx].bookmaker

        if all_odds_x:
            consensus.avg_odds_x = sum(all_odds_x) / len(all_odds_x)
            consensus.best_odds_x = max(all_odds_x)
            best_idx = all_odds_x.index(max(all_odds_x))
            consensus.best_bookmaker_x = consensus.snapshots[best_idx].bookmaker

        if all_odds_2:
            consensus.avg_odds_2 = sum(all_odds_2) / len(all_odds_2)
            consensus.best_odds_2 = max(all_odds_2)
            best_idx = all_odds_2.index(max(all_odds_2))
            consensus.best_bookmaker_2 = consensus.snapshots[best_idx].bookmaker

        # Calculer les probabilités consensus (sans marge)
        if consensus.avg_odds_1 > 0 and consensus.avg_odds_x > 0 and consensus.avg_odds_2 > 0:
            raw = [
                1/consensus.avg_odds_1,
                1/consensus.avg_odds_x,
                1/consensus.avg_odds_2
            ]
            total = sum(raw)
            consensus.consensus_prob_1 = raw[0] / total
            consensus.consensus_prob_x = raw[1] / total
            consensus.consensus_prob_2 = raw[2] / total

        # Probabilités Pinnacle (benchmark le plus fiable)
        if consensus.pinnacle_odds_1 > 0:
            raw = [
                1/consensus.pinnacle_odds_1,
                1/consensus.pinnacle_odds_x,
                1/consensus.pinnacle_odds_2
            ]
            total = sum(raw)
            consensus.pinnacle_prob_1 = raw[0] / total
            consensus.pinnacle_prob_x = raw[1] / total
            consensus.pinnacle_prob_2 = raw[2] / total

        # Marge moyenne
        margins = [s.margin for s in consensus.snapshots if s.margin > 0]
        consensus.avg_margin = sum(margins) / max(len(margins), 1)

        return consensus

    # ─── ANALYSE DES COTES BETCLIC ──────────────────

    def compare_betclic_to_market(self, betclic_odds: Dict,
                                   consensus: MarketConsensus) -> Dict:
        """
        Compare les cotes Betclic au consensus du marché.

        Identifie si Betclic offre des cotes supérieures ou
        inférieures au marché pour chaque sélection.
        """

        result = {
            "betclic_vs_market": {},
            "betclic_vs_pinnacle": {},
            "betclic_vs_best": {},
        }

        selections = {
            "1": (betclic_odds.get("1", 0), consensus.avg_odds_1,
                  consensus.pinnacle_odds_1, consensus.best_odds_1,
                  consensus.best_bookmaker_1),
            "X": (betclic_odds.get("X", 0), consensus.avg_odds_x,
                  consensus.pinnacle_odds_x, consensus.best_odds_x,
                  consensus.best_bookmaker_x),
            "2": (betclic_odds.get("2", 0), consensus.avg_odds_2,
                  consensus.pinnacle_odds_2, consensus.best_odds_2,
                  consensus.best_bookmaker_2),
        }

        for sel, (bc, market_avg, pinnacle, best, best_bk) in selections.items():
            if bc > 0 and market_avg > 0:
                # Betclic vs Marché
                diff_market = ((bc - market_avg) / market_avg) * 100
                result["betclic_vs_market"][sel] = {
                    "betclic": bc,
                    "market_avg": round(market_avg, 2),
                    "diff_pct": round(diff_market, 2),
                    "status": "ABOVE" if diff_market > 0 else "BELOW"
                }

                # Betclic vs Pinnacle
                if pinnacle > 0:
                    diff_pin = ((bc - pinnacle) / pinnacle) * 100
                    result["betclic_vs_pinnacle"][sel] = {
                        "betclic": bc,
                        "pinnacle": pinnacle,
                        "diff_pct": round(diff_pin, 2),
                        "status": "ABOVE" if diff_pin > 0 else "BELOW"
                    }

                # Betclic vs Meilleure cote
                if best > 0:
                    result["betclic_vs_best"][sel] = {
                        "betclic": bc,
                        "best": best,
                        "best_bookmaker": best_bk,
                        "loss_pct": round(((best - bc) / best) * 100, 2)
                    }

        return result
