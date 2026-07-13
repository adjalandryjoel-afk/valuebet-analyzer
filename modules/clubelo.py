"""
═══════════════════════════════════════════════════════
 MODULE CLUBELO — Ratings Elo réels via clubelo.com
═══════════════════════════════════════════════════════

Fournit au modèle Elo des ratings INDÉPENDANTS des cotes :
ClubElo publie chaque jour l'Elo de tous les clubs européens,
gratuitement et sans clé API.

  GET http://api.clubelo.com/{YYYY-MM-DD}
  → CSV : Rank,Club,Country,Level,Elo,From,To
    (tous les clubs du jour en UNE seule requête)

Économie de requêtes :
  • snapshot du jour téléchargé une fois puis mis en cache
    disque data/clubelo_cache.json, TTL 3 jours
  • pendant l'intersaison, certains clubs (ex : Bayern)
    disparaissent du CSV du jour entre deux périodes de
    rating → un snapshot de complément (~3 semaines en
    arrière) comble ces trous (2 requêtes max tous les 3 j)
  • rechargé en mémoire à l'instanciation
  • échec réseau → le cache périmé reste accepté, sinon
    None partout

Matching des noms (ClubElo utilise des noms courts :
"Paris SG", "Man City", "Bayern", "Inter", "Dortmund"...) :
  1. alias explicites pour les pièges connus
  2. correspondance exacte (noms normalisés)
  3. inclusion de tokens (ex : "marseille" ⊂
     "olympique de marseille")
  4. rapidfuzz token_sort_ratio ≥ 75

Toute erreur (réseau, CSV illisible, équipe introuvable)
retourne None sans lever d'exception.
"""

import os
import re
import csv
import io
import json
import time
import unicodedata
import requests
from datetime import date, timedelta
from typing import Dict, Optional

from config import Paths

try:
    from rapidfuzz import fuzz, process as rf_process
except ImportError:  # pragma: no cover
    fuzz = rf_process = None


# ══════════════════════════════════════════════════════
#  FOURNISSEUR CLUBELO
# ══════════════════════════════════════════════════════

class ClubEloProvider:
    """
    Fournit le rating Elo ClubElo d'une équipe par son nom.

    Échelle ClubElo : ~1900 pour un top club européen, ~1500
    pour un club moyen de grand championnat. INCOMPATIBLE avec
    les ratings estimés depuis les cotes (centrés sur 1500) :
    ne jamais mélanger les deux échelles dans une même prédiction.
    """

    BASE_URL = "http://api.clubelo.com"

    # Durée de vie du snapshot en cache (secondes) — les ratings
    # ClubElo bougent peu d'un jour à l'autre
    CACHE_TTL = 3 * 24 * 3600

    # Recul du snapshot de complément (jours) : comble les clubs
    # absents du CSV du jour entre deux périodes de rating
    # (intersaison) — leur rating est de toute façon figé
    BACKFILL_DAYS = 21

    # Score rapidfuzz minimum pour accepter une correspondance floue
    FUZZY_THRESHOLD = 75

    # Alias explicites : nom normalisé Betclic → nom normalisé ClubElo
    # (pièges connus où ni l'exact ni l'inclusion ne suffisent)
    ALIASES = {
        "paris saint germain": "paris sg",
        "psg": "paris sg",
        "manchester city": "man city",
        "manchester united": "man united",
        "bayern munich": "bayern",
        "bayern munchen": "bayern",
        "borussia dortmund": "dortmund",
        "inter milan": "inter",
        "inter milan fc": "inter",
        "ac milan": "milan",
        "milan ac": "milan",
        "atletico madrid": "atletico",
        "atletico de madrid": "atletico",
        "olympique de marseille": "marseille",
        "olympique marseille": "marseille",
        "olympique lyonnais": "lyon",
        "olympique lyon": "lyon",
        "tottenham hotspur": "tottenham",
        "athletic bilbao": "bilbao",
        "athletic club": "bilbao",
        "fc barcelone": "barcelona",
        "barcelone": "barcelona",
        "juventus turin": "juventus",
        "bayer leverkusen": "leverkusen",
        "borussia monchengladbach": "gladbach",
        "naples": "napoli",
        "seville fc": "sevilla",
        "fc seville": "sevilla",
        "real betis seville": "betis",
        "west ham united": "west ham",
        "wolverhampton wanderers": "wolves",
        "wolverhampton": "wolves",
        "newcastle united": "newcastle",
        "leeds united": "leeds",
        "leicester city": "leicester",
        "sporting lisbonne": "sporting",
        "sporting cp": "sporting",
        "benfica lisbonne": "benfica",
        "ajax amsterdam": "ajax",
        "celtic glasgow": "celtic",
    }

    def __init__(self):
        self.cache_path = os.path.join(Paths.DATA_DIR, "clubelo_cache.json")

        # ratings : nom ClubElo normalisé → Elo
        self.ratings: Dict[str, float] = {}

        # Cache mémoire des résolutions de noms (échecs compris) :
        # nom normalisé demandé → Elo ou None
        self._resolved: Dict[str, Optional[float]] = {}

        self._load()

    # ─── NORMALISATION (même logique qu'api_football) ───

    @staticmethod
    def _normalize(team_name: str) -> str:
        """
        Normalise un nom d'équipe : minuscules, sans accents ni
        ponctuation (même logique qu'ApiFootballCollector._normalize).
        """

        name = unicodedata.normalize("NFKD", team_name or "")
        name = "".join(c for c in name if not unicodedata.combining(c))
        name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
        return " ".join(name.split())

    # ─── CACHE DISQUE ───────────────────────────────

    def _read_cache(self) -> Optional[Dict]:
        """Lit le cache disque (absent/corrompu → None)."""

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(
                    data.get("ratings"), dict):
                return data
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return None

    def _write_cache(self, snapshot_date: str, ratings: Dict[str, float]):
        """Sauvegarde le snapshot sur disque (échec silencieux)."""

        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "date": snapshot_date,
                    "fetched_ts": time.time(),
                    "ratings": ratings,
                }, f, ensure_ascii=False, indent=2)
        except (OSError, TypeError, ValueError):
            pass

    # ─── TÉLÉCHARGEMENT DU SNAPSHOT ─────────────────

    def _fetch_snapshot(self, day: date,
                        quiet: bool = False) -> Optional[Dict[str, float]]:
        """
        Télécharge le snapshot ClubElo d'un jour donné (tous les
        clubs européens en une requête). Retourne {nom_normalisé:
        elo}, ou None en cas d'échec. Ne lève jamais d'exception.
        """

        try:
            response = requests.get(
                f"{self.BASE_URL}/{day.isoformat()}", timeout=15)
        except requests.RequestException as e:
            if not quiet:
                print(f"      ⚠️ ClubElo indisponible : {e}")
            return None

        if response.status_code != 200:
            if not quiet:
                print(f"      ⚠️ ClubElo indisponible : "
                      f"HTTP {response.status_code}")
            return None

        ratings: Dict[str, float] = {}

        try:
            reader = csv.DictReader(io.StringIO(response.text))
            for row in reader:
                name = self._normalize(row.get("Club") or "")
                if not name:
                    continue
                elo = float(row.get("Elo"))
                # Le CSV est trié par Elo décroissant : en cas de
                # collision de noms normalisés, garder le plus fort
                if name not in ratings:
                    ratings[name] = round(elo, 1)
        except (ValueError, TypeError, KeyError, csv.Error):
            if not quiet:
                print("      ⚠️ ClubElo indisponible : CSV illisible")
            return None

        if not ratings:
            if not quiet:
                print("      ⚠️ ClubElo indisponible : snapshot vide")
            return None

        return ratings

    def _load(self):
        """
        Charge les ratings en mémoire : cache disque frais (TTL 3 j)
        en priorité, sinon téléchargement (snapshot du jour, complété
        par un snapshot antérieur pour les clubs en trou d'intersaison),
        sinon cache périmé.
        """

        cached = self._read_cache()

        # 1. Cache frais → zéro requête
        if cached and (time.time() - cached.get("fetched_ts", 0)) \
                < self.CACHE_TTL:
            self.ratings = cached["ratings"]
            return

        # 2. Téléchargement du snapshot du jour
        today = date.today()
        fresh = self._fetch_snapshot(today)
        if fresh:
            # Complément : clubs absents du jour (entre deux périodes
            # de rating, ex : Bayern à l'intersaison)
            backfill = self._fetch_snapshot(
                today - timedelta(days=self.BACKFILL_DAYS), quiet=True)
            if backfill:
                for name, elo in backfill.items():
                    fresh.setdefault(name, elo)

            print(f"      📡 ClubElo : snapshot du {today.isoformat()} "
                  f"({len(fresh)} clubs)")
            self.ratings = fresh
            self._write_cache(today.isoformat(), fresh)
            return

        # 3. Échec réseau → cache périmé accepté, sinon vide
        if cached:
            print("      ⚠️ ClubElo : cache périmé utilisé en secours")
            self.ratings = cached["ratings"]

    # ─── MATCHING DES NOMS ──────────────────────────

    def _match(self, wanted: str) -> Optional[float]:
        """
        Résout un nom normalisé vers un Elo ClubElo :
        alias → exact → inclusion de tokens → fuzzy (≥ 75).
        """

        # 1. Alias explicites (pièges connus)
        wanted = self.ALIASES.get(wanted, wanted)

        # 2. Correspondance exacte
        if wanted in self.ratings:
            return self.ratings[wanted]

        # 3. Inclusion : les tokens d'un nom sont un sous-ensemble
        #    de l'autre (ex : "marseille" ⊂ "olympique de marseille"),
        #    avec au moins un token significatif (≥ 4 lettres)
        wanted_tokens = set(wanted.split())
        best_name, best_score = None, 0.0

        for name in self.ratings:
            name_tokens = set(name.split())
            small = name_tokens if len(name_tokens) <= len(wanted_tokens) \
                else wanted_tokens
            big = wanted_tokens if small is name_tokens else name_tokens
            if not small or not small <= big:
                continue
            if not any(len(t) >= 4 for t in small):
                continue
            score = fuzz.token_sort_ratio(wanted, name) if fuzz \
                else 100.0
            if score > best_score:
                best_name, best_score = name, score

        if best_name:
            return self.ratings[best_name]

        # 4. Correspondance floue rapidfuzz
        if rf_process and self.ratings:
            found = rf_process.extractOne(
                wanted, self.ratings.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.FUZZY_THRESHOLD,
            )
            if found:
                return self.ratings[found[0]]

        return None

    # ─── POINT D'ENTRÉE ─────────────────────────────

    def get_rating(self, team_name: str) -> Optional[float]:
        """
        Retourne l'Elo ClubElo d'une équipe, ou None si elle est
        introuvable (club non européen, nom irrésoluble) ou si le
        snapshot n'a pas pu être chargé. Ne lève jamais d'exception.
        """

        if not team_name or not team_name.strip() or not self.ratings:
            return None

        key = self._normalize(team_name)
        if not key:
            return None

        # Cache mémoire des résolutions (échecs compris)
        if key in self._resolved:
            return self._resolved[key]

        rating = self._match(key)
        self._resolved[key] = rating
        return rating
