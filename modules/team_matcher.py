"""
═══════════════════════════════════════════════════════
 MODULE TEAM MATCHER — Identification des équipes
 depuis les noms bruts extraits des captures Betclic
═══════════════════════════════════════════════════════

Les noms d'équipes sur Betclic sont souvent abrégés ou
mal orthographiés ("Man City", "PSG", "Asec"...). Ce
module les rattache aux noms officiels via fuzzy matching
sur la base data/teams_database.json.
"""

import os
import json
from typing import Dict, Optional

from config import Paths

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False


class TeamMatcher:
    """Rattache un nom brut d'équipe à son identité officielle."""

    FUZZY_CUTOFF = 70  # score minimum (0-100) pour accepter un match

    def __init__(self):
        self.teams: Dict[str, Dict] = {}
        self._alias_index: Dict[str, str] = {}
        self._load_database()

    def _load_database(self):
        """Charge la base d'équipes et construit l'index des alias."""

        if os.path.exists(Paths.TEAMS_DATABASE):
            try:
                with open(Paths.TEAMS_DATABASE, 'r', encoding='utf-8') as f:
                    self.teams = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.teams = {}

        for official, info in self.teams.items():
            self._alias_index[official.lower()] = official
            for alias in info.get("aliases", []):
                self._alias_index[alias.lower()] = official

    # ─── IDENTIFICATION ─────────────────────────────

    def identify_team(self, raw_name: str) -> Dict:
        """
        Identifie une équipe depuis son nom brut.

        Retourne : {
            "raw_name": nom d'origine,
            "official_name": nom officiel (ou nom brut nettoyé),
            "matched": True/False,
            "score": score de confiance du matching,
            "info": fiche de l'équipe (league, city...) ou None
        }
        """

        cleaned = raw_name.strip()

        if not cleaned:
            return {
                "raw_name": raw_name,
                "official_name": raw_name,
                "matched": False,
                "score": 0,
                "info": None,
            }

        # 1. Match exact sur nom officiel ou alias
        official = self._alias_index.get(cleaned.lower())
        if official:
            return {
                "raw_name": raw_name,
                "official_name": official,
                "matched": True,
                "score": 100,
                "info": self.teams.get(official),
            }

        # 2. Fuzzy matching
        candidates = list(self._alias_index.keys())
        if candidates:
            if HAS_RAPIDFUZZ:
                match = process.extractOne(
                    cleaned.lower(),
                    candidates,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=self.FUZZY_CUTOFF,
                )
                if match:
                    alias, score = match[0], match[1]
                    official = self._alias_index[alias]
                    return {
                        "raw_name": raw_name,
                        "official_name": official,
                        "matched": True,
                        "score": round(score, 1),
                        "info": self.teams.get(official),
                    }
            else:
                close = difflib.get_close_matches(
                    cleaned.lower(), candidates, n=1,
                    cutoff=self.FUZZY_CUTOFF / 100,
                )
                if close:
                    official = self._alias_index[close[0]]
                    return {
                        "raw_name": raw_name,
                        "official_name": official,
                        "matched": True,
                        "score": 80.0,
                        "info": self.teams.get(official),
                    }

        # 3. Équipe inconnue : garder le nom nettoyé
        return {
            "raw_name": raw_name,
            "official_name": cleaned,
            "matched": False,
            "score": 0,
            "info": None,
        }

    def identify_match(self, home_raw: str, away_raw: str) -> Dict:
        """
        Identifie les deux équipes d'un match et infère la ligue.

        Retourne : {
            "home": {...}, "away": {...},
            "league": clé de ligue ou "unknown",
            "both_matched": bool
        }
        """

        home = self.identify_team(home_raw)
        away = self.identify_team(away_raw)

        # Inférer la ligue
        home_league = (home["info"] or {}).get("league")
        away_league = (away["info"] or {}).get("league")

        if home_league and home_league == away_league:
            league = home_league
        elif home_league and not away_league:
            league = home_league
        elif away_league and not home_league:
            league = away_league
        elif home_league and away_league:
            # Ligues différentes → probablement une coupe
            league = "champions_league"
        else:
            league = "unknown"

        return {
            "home": home,
            "away": away,
            "league": league,
            "both_matched": home["matched"] and away["matched"],
        }
