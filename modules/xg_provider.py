"""
═══════════════════════════════════════════════════════
 MODULE XG PROVIDER — Profils xG via soccerdata/Understat
═══════════════════════════════════════════════════════

Fournit les moyennes xG par match (pour/contre, domicile/
extérieur) d'une équipe des 5 grands championnats, sur la
saison la plus récente disponible chez Understat.

Remplace le scraping xG maison (bloqué : Understat 403 /
format changé) en s'appuyant sur la librairie communautaire
soccerdata, qui externalise la maintenance du parsing.

Double cache :
1. soccerdata gère son propre cache disque (le premier
   chargement d'une ligue prend 1-3 min, quasi instantané
   ensuite) ;
2. ce module maintient data/xg_cache.json avec les profils
   déjà calculés (TTL 24h, les échecs sont aussi mémorisés
   24h pour ne pas retenter en boucle).

L'import de soccerdata est PARESSEUX : si la librairie est
absente ou cassée, le module retourne None sans jamais
empêcher l'application de démarrer.
"""

import os
import json
import time
import logging
import unicodedata
from typing import Dict, Optional

from rapidfuzz import process, fuzz

from config import Paths


class XgProvider:
    """Profils xG d'équipes via soccerdata (source Understat)."""

    # Ligues de l'app → identifiants soccerdata/Understat.
    # Toute autre ligue → None direct (aucune requête).
    LEAGUE_MAP = {
        "premier_league": "ENG-Premier League",
        "la_liga": "ESP-La Liga",
        "serie_a": "ITA-Serie A",
        "bundesliga": "GER-Bundesliga",
        "ligue1_fr": "FRA-Ligue 1",
    }

    # Saisons essayées dans l'ordre (2025 = saison 2025-26)
    SEASONS = (2025, 2024)

    # Cache local des profils calculés
    CACHE_TTL_SECONDS = 24 * 3600

    # Score rapidfuzz minimum pour accepter une correspondance de nom
    FUZZY_THRESHOLD = 70

    # Au-delà de cette ancienneté du dernier match, le profil
    # est jugé périmé et n'est pas utilisé
    MAX_STALENESS_DAYS = 550

    def __init__(self):
        self.cache_path = os.path.join(Paths.DATA_DIR, "xg_cache.json")
        self._cache = self._load_cache()
        # DataFrames de ligue déjà chargés pendant cette session
        # clé : (ligue_soccerdata, saison)
        self._frames: Dict = {}

    # ─── API PUBLIQUE ───────────────────────────────

    def get_xg_profile(self, team_name: str,
                       league: str) -> Optional[Dict]:
        """
        Profil xG moyen par match d'une équipe sur la saison la
        plus récente disponible.

        Retourne un dict :
          {"xg_for_avg", "xga_avg", "xg_for_home", "xga_home",
           "xg_for_away", "xga_away", "matches", "season",
           "last_match_date"}
        ou None (ligue non couverte, équipe introuvable, données
        périmées, librairie indisponible...). Ne lève jamais.
        """

        sd_league = self.LEAGUE_MAP.get(league)
        if not sd_league:
            return None  # ligue non couverte : pas de requête

        key = f"{self._normalize(team_name)}|{league}"

        entry = self._cache.get(key)
        if entry and time.time() - entry.get("ts", 0) < self.CACHE_TTL_SECONDS:
            return entry.get("profile")  # peut être None (échec mémorisé)

        try:
            profile = self._compute_profile(team_name, sd_league)
        except Exception:
            profile = None

        # Succès comme échec sont mis en cache 24h
        self._cache[key] = {"ts": time.time(), "profile": profile}
        self._save_cache()

        return profile

    # ─── CALCUL DU PROFIL ───────────────────────────

    def _compute_profile(self, team_name: str,
                         sd_league: str) -> Optional[Dict]:
        """Essaie chaque saison (2025 puis 2024) jusqu'à trouver l'équipe."""

        for season in self.SEASONS:
            df = self._load_league(sd_league, season)
            if df is None or df.empty:
                continue

            understat_name = self._match_team_name(team_name, df)
            if not understat_name:
                continue

            profile = self._profile_from_matches(df, understat_name, season)
            if profile:
                return profile

        return None

    def _load_league(self, sd_league: str, season: int):
        """
        Charge les stats de matchs d'une ligue/saison via soccerdata.

        Import paresseux : la librairie ne doit jamais empêcher
        l'app de démarrer si elle est absente ou cassée.
        Le premier chargement d'une ligue peut prendre 1-3 min
        (cache disque natif de soccerdata ensuite).
        """

        cache_key = (sd_league, season)
        if cache_key in self._frames:
            ts, frame = self._frames[cache_key]
            if __import__("time").time() - ts < 24 * 3600:
                return frame
            del self._frames[cache_key]

        try:
            import soccerdata as sd
            logging.getLogger("soccerdata").setLevel(logging.WARNING)

            understat = sd.Understat(leagues=sd_league, seasons=season)
            df = understat.read_team_match_stats()

            # Ne garder que les matchs joués (xG renseigné)
            df = df[df["home_xg"].notna() & df["away_xg"].notna()]
        except Exception:
            df = None

        self._frames[cache_key] = (__import__("time").time(), df)
        return df

    def _match_team_name(self, team_name: str, df) -> Optional[str]:
        """Nom Understat le plus proche du nom de l'app (rapidfuzz)."""

        candidates = sorted(
            set(df["home_team"].dropna()) | set(df["away_team"].dropna())
        )
        if not candidates:
            return None

        result = process.extractOne(
            self._normalize(team_name),
            candidates,
            scorer=fuzz.WRatio,
            processor=self._normalize,
            score_cutoff=self.FUZZY_THRESHOLD,
        )
        return result[0] if result else None

    def _profile_from_matches(self, df, team: str,
                              season: int) -> Optional[Dict]:
        """Moyennes xG par match depuis les lignes de matchs de l'équipe."""

        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]

        n_home, n_away = len(home), len(away)
        matches = n_home + n_away
        if matches == 0:
            return None

        # Fraîcheur : dernier match trop ancien → profil périmé
        last_date = max(home["date"].max(), away["date"].max()) \
            if (n_home and n_away) \
            else (home["date"].max() if n_home else away["date"].max())

        age_days = (self._now() - last_date.to_pydatetime()).days
        if age_days > self.MAX_STALENESS_DAYS:
            return None

        xg_home_for = float(home["home_xg"].sum())
        xg_home_against = float(home["away_xg"].sum())
        xg_away_for = float(away["away_xg"].sum())
        xg_away_against = float(away["home_xg"].sum())

        return {
            "xg_for_avg": round((xg_home_for + xg_away_for) / matches, 3),
            "xga_avg": round((xg_home_against + xg_away_against) / matches, 3),
            "xg_for_home": round(xg_home_for / n_home, 3) if n_home else None,
            "xga_home": round(xg_home_against / n_home, 3) if n_home else None,
            "xg_for_away": round(xg_away_for / n_away, 3) if n_away else None,
            "xga_away": round(xg_away_against / n_away, 3) if n_away else None,
            "matches": matches,
            "season": f"{season}-{(season + 1) % 100:02d}",
            "last_match_date": last_date.strftime("%Y-%m-%d"),
        }

    # ─── HELPERS ────────────────────────────────────

    @staticmethod
    def _now():
        from datetime import datetime
        return datetime.now()

    @staticmethod
    def _normalize(name: str) -> str:
        """Minuscules, sans accents ni ponctuation parasite."""

        name = unicodedata.normalize("NFKD", str(name))
        name = "".join(c for c in name if not unicodedata.combining(c))
        return name.lower().replace("-", " ").replace(".", " ").strip()

    # ─── CACHE LOCAL (data/xg_cache.json) ───────────

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # cache non persisté : non bloquant
