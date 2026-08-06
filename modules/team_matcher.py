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

    # ── APPARIEMENT FLOU : trois verrous, pas un seuil ──
    #
    # Un seuil unique de 70 sur une base de 45 equipes appariait
    # n'importe quoi. Cas reel, argent reel : « SC Braga » (Ligue
    # Conference) est devenu « SC Gagnoa » (Ligue 1 ivoirienne) avec
    # un score de 70,6 — un dixieme de point au-dessus du seuil. Et
    # « Sporting Braga » -> « Sporting Gagnoa » atteignait 83.
    # L'analyse entiere partait ensuite sur le mauvais club : mauvais
    # championnat, mauvaises statistiques, mauvais pari.
    #
    # Le mot GENERIQUE (sc, sporting, real...) faisait tout le score.
    # Le discriminant, c'est le mot DISTINCTIF : « braga » contre
    # « gagnoa » ne vaut que 55.
    FUZZY_CUTOFF = 85          # score global minimum
    FUZZY_COEUR_MIN = 80       # score du mot distinctif, hors generiques
    FUZZY_MARGE = 8            # avance minimale sur le 2e candidat

    # Mots de club sans pouvoir discriminant : deux clubs de pays
    # differents les partagent constamment.
    MOTS_GENERIQUES = {
        "sc", "fc", "ac", "as", "ss", "us", "cd", "ca", "cf", "sk",
        "sv", "fk", "nk", "hk", "if", "ik", "bk", "afc", "cfc",
        "sporting", "club", "real", "atletico", "athletic",
        "deportivo", "united", "city", "olympique", "racing",
        "stade", "sport", "sports", "de", "du", "la", "le", "of",
    }

    @classmethod
    def _coeur(cls, nom: str) -> str:
        """Le nom prive de ses mots generiques — sa partie distinctive."""

        mots = [m for m in nom.lower().split()
                if m not in cls.MOTS_GENERIQUES]
        return " ".join(mots)

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
                tops = process.extract(
                    cleaned.lower(), candidates,
                    scorer=fuzz.token_sort_ratio, limit=2)
                alias = score = None
                if tops and tops[0][1] >= self.FUZZY_CUTOFF:
                    alias, score = tops[0][0], tops[0][1]
                    # Verrou 1 : avance nette sur le second. Deux
                    # candidats a egalite = on ne sait pas, donc on
                    # refuse.
                    if (len(tops) > 1
                            and score - tops[1][1] < self.FUZZY_MARGE):
                        alias = None
                    # Verrou 2 : le mot DISTINCTIF doit correspondre.
                    # C'est lui qui separe « braga » de « gagnoa ».
                    if alias is not None:
                        ca = self._coeur(cleaned)
                        cb = self._coeur(alias)
                        if ca and cb and fuzz.token_sort_ratio(
                                ca, cb) < self.FUZZY_COEUR_MIN:
                            alias = None
                if alias is not None:
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
