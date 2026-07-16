"""
═══════════════════════════════════════════════════════
 MODULE DAILY SCANNER — matchs du jour, sans capture
═══════════════════════════════════════════════════════

Deux étapes, pour dépenser le quota The Odds API au minimum :

  1. list_fixtures()  → matchs à venir des ligues suivies EN
     SAISON. Endpoint /events : GRATUIT (0 crédit). Sert à voir
     le programme et à choisir quoi analyser.

  2. scan_league()    → cotes 1X2 + Over/Under 2.5 de TOUS les
     matchs d'une ligue en une requête (~2 crédits), moyennées
     sur les bookmakers. De quoi lancer une pré-analyse complète
     (mêmes modèles que les captures) sur tout un programme.

LIMITE ASSUMÉE : The Odds API ne fournit que le 1X2 et l'Over/Under
2.5 — pas les marchés riches de Betclic (tirs cadrés, buts par
équipe/mi-temps). Le scan sert donc à REPÉRER la value du jour ;
pour l'analyse complète d'un match, la capture reste supérieure.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from config import APIKeys
from modules.odds_collector import OddsAPICollector

BASE_URL = "https://api.the-odds-api.com/v4"
# clé de ligue interne → nom lisible (repris de SUPPORTED_LEAGUES)
_SPORT_TO_LEAGUE = {v: k for k, v in OddsAPICollector.LEAGUE_KEYS.items()}


class DailyScanner:
    """Programme du jour et cotes de base via The Odds API."""

    def __init__(self):
        self.api_key = APIKeys.ODDS_API_KEY
        self.quota_restant: Optional[str] = None

    # ─── LIGUES EN SAISON (gratuit) ──────────────────

    def in_season_leagues(self) -> Dict[str, str]:
        """
        {clé_interne: sport_key} des ligues suivies actuellement en
        saison. /sports est GRATUIT : évite d'interroger une ligue
        à l'arrêt.
        """
        suivies = OddsAPICollector.LEAGUE_KEYS
        try:
            r = requests.get(f"{BASE_URL}/sports",
                             params={"apiKey": self.api_key}, timeout=20)
            self.quota_restant = r.headers.get("x-requests-remaining")
            if r.status_code != 200:
                return dict(suivies)
            actives = {s["key"] for s in r.json() if s.get("active")}
        except (requests.RequestException, ValueError, KeyError):
            return dict(suivies)

        return {lg: sk for lg, sk in suivies.items() if sk in actives}

    # ─── PROGRAMME À VENIR (gratuit) ─────────────────

    def list_fixtures(self, sport_key: str,
                      days_ahead: int = 3) -> List[Dict]:
        """
        Matchs à venir d'une ligue dans les `days_ahead` prochains
        jours. GRATUIT (endpoint /events, 0 crédit).
        """
        if not self.api_key:
            return []
        try:
            r = requests.get(f"{BASE_URL}/sports/{sport_key}/events",
                             params={"apiKey": self.api_key}, timeout=20)
            self.quota_restant = r.headers.get("x-requests-remaining")
            if r.status_code != 200:
                return []
            events = r.json()
        except (requests.RequestException, ValueError):
            return []

        now = datetime.now(timezone.utc)
        fixtures = []
        for e in events:
            ko = self._parse_ko(e.get("commence_time"))
            if ko is None:
                continue
            delta_j = (ko - now).total_seconds() / 86400.0
            if 0 <= delta_j <= days_ahead:
                fixtures.append({
                    "home": e.get("home_team"),
                    "away": e.get("away_team"),
                    "kickoff": ko,
                    "sport_key": sport_key,
                    "league": _SPORT_TO_LEAGUE.get(sport_key, ""),
                })
        return sorted(fixtures, key=lambda f: f["kickoff"])

    # ─── COTES D'UNE LIGUE (≈ 2 crédits) ─────────────

    def scan_league(self, sport_key: str,
                    days_ahead: int = 3) -> List[Dict]:
        """
        Cotes 1X2 + Over/Under 2.5 de tous les matchs à venir d'une
        ligue, moyennées sur les bookmakers. Une seule requête
        (~2 crédits). Format prêt pour analyze_matches_ui.
        """
        if not self.api_key:
            return []
        try:
            # Coût The Odds API = nb marchés × nb régions. On garde 1
            # seule région (eu, déjà ~20 bookmakers) → 2 marchés × 1 =
            # 2 crédits par championnat, comme annoncé à l'utilisateur.
            r = requests.get(f"{BASE_URL}/sports/{sport_key}/odds",
                             params={"apiKey": self.api_key,
                                     "regions": "eu",
                                     "markets": "h2h,totals",
                                     "oddsFormat": "decimal"},
                             timeout=30)
            self.quota_restant = r.headers.get("x-requests-remaining")
            if r.status_code != 200:
                return []
            data = r.json()
        except (requests.RequestException, ValueError):
            return []

        now = datetime.now(timezone.utc)
        league = _SPORT_TO_LEAGUE.get(sport_key, "")
        matches = []
        for m in data:
            ko = self._parse_ko(m.get("commence_time"))
            if ko is None or not (0 <= (ko - now).total_seconds()
                                  / 86400.0 <= days_ahead):
                continue

            home, away = m.get("home_team"), m.get("away_team")
            cotes, extras = self._extract(m, home, away)
            if not (cotes.get("1") and cotes.get("X") and cotes.get("2")):
                continue

            matches.append({
                "equipe_domicile": home,
                "equipe_exterieur": away,
                "competition": league,
                "kickoff": ko,
                "cotes": cotes,
                "marches_supplementaires": extras,
            })
        return sorted(matches, key=lambda x: x["kickoff"])

    # ─── OUTILS ──────────────────────────────────────

    @staticmethod
    def _extract(match: Dict, home: str, away: str):
        """Moyenne bookmakers du 1X2 et de l'Over/Under 2.5."""
        h2h: Dict[str, List[float]] = {}
        ou: Dict[str, List[float]] = {}
        for b in match.get("bookmakers", []):
            for mk in b.get("markets", []):
                if mk.get("key") == "h2h":
                    for o in mk.get("outcomes", []):
                        nom = o.get("name")
                        cle = ("1" if nom == home
                               else "2" if nom == away else "X")
                        try:
                            h2h.setdefault(cle, []).append(float(o["price"]))
                        except (KeyError, TypeError, ValueError):
                            pass
                elif mk.get("key") == "totals":
                    for o in mk.get("outcomes", []):
                        try:
                            if abs(float(o.get("point", 0)) - 2.5) > 1e-9:
                                continue
                            k = ("over_2_5" if o.get("name") == "Over"
                                 else "under_2_5")
                            ou.setdefault(k, []).append(float(o["price"]))
                        except (KeyError, TypeError, ValueError):
                            pass

        def moy(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0

        cotes = {k: moy(h2h.get(k, [])) for k in ("1", "X", "2")}
        extras = {k: moy(ou.get(k, [])) for k in ("over_2_5", "under_2_5")
                  if ou.get(k)}
        return cotes, extras

    @staticmethod
    def _parse_ko(iso: Optional[str]) -> Optional[datetime]:
        if not iso:
            return None
        try:
            return datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        except ValueError:
            return None
