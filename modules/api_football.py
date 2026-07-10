"""
═══════════════════════════════════════════════════════
 MODULE API-FOOTBALL — Statistiques réelles des équipes
 via api-sports.io (accès direct, plan gratuit)
═══════════════════════════════════════════════════════

Fournit au modèle des statistiques INDÉPENDANTES des cotes :
buts marqués/encaissés (globaux et domicile/extérieur) et
forme récente, calculés sur les 15 derniers matchs terminés.

Économie de requêtes (plan gratuit : 100 requêtes/jour) :
  • cache disque data/api_cache.json
  • team_id conservé indéfiniment (1 seule recherche/équipe)
  • stats conservées 24h (TTL)
  • saison accessible mémorisée (le plan gratuit n'accepte ni
    le paramètre `last` ni les saisons trop récentes)

Toute erreur (clé absente, HTTP != 200, quota épuisé, réseau)
retourne None sans lever d'exception : le DataCollector
bascule alors sur ses sources de repli.
"""

import os
import re
import json
import time
import unicodedata
import requests
from datetime import datetime
from typing import Dict, List, Optional

from config import APIKeys, Paths


# ══════════════════════════════════════════════════════
#  COLLECTEUR API-FOOTBALL
# ══════════════════════════════════════════════════════

class ApiFootballCollector:
    """
    Collecte les stats d'une équipe via l'API api-sports.io (v3).

    Enchaînement (2 requêtes max par équipe, puis 0 grâce au cache) :
    1. GET /teams?search={nom}          → team_id  (mis en cache à vie)
    2. GET /fixtures?team={id}&season=S → 15 derniers matchs terminés
       → moyennes de buts, splits domicile/extérieur, forme récente
       (mis en cache 24h)

    Le plan gratuit refuse le paramètre `last` et les saisons trop
    récentes : on interroge donc la saison la plus récente autorisée
    (découverte via le message d'erreur du plan, puis mémorisée dans
    le cache sous la clé "_meta") et on filtre côté client.
    """

    BASE_URL = "https://v3.football.api-sports.io"

    # Durée de vie des stats en cache (secondes)
    CACHE_TTL = 24 * 3600

    # Nombre de derniers matchs terminés utilisés
    LAST_FIXTURES = 15

    # Statuts de matchs considérés comme terminés
    FINISHED_STATUSES = ("FT", "AET", "PEN")

    def __init__(self):
        self.api_key = APIKeys.RAPIDAPI_KEY
        self.cache_path = os.path.join(Paths.DATA_DIR, "api_cache.json")
        self.cache: Dict = self._load_cache()
        self._request_count = 0

    # ─── CACHE DISQUE ───────────────────────────────

    def _load_cache(self) -> Dict:
        """Charge le cache disque (fichier absent/corrompu → cache vide)."""

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

    def _save_cache(self):
        """Sauvegarde le cache sur disque (échec silencieux)."""

        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except (OSError, TypeError, ValueError):
            pass

    @staticmethod
    def _normalize(team_name: str) -> str:
        """
        Normalise un nom d'équipe : minuscules, sans accents ni
        ponctuation. Sert de clé de cache ET de requête de recherche
        (l'API n'accepte que lettres, chiffres et espaces).
        """

        name = unicodedata.normalize("NFKD", team_name or "")
        name = "".join(c for c in name if not unicodedata.combining(c))
        name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
        return " ".join(name.split())

    # ─── REQUÊTES HTTP ──────────────────────────────

    def _request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Exécute une requête GET sur l'API.

        Retourne le JSON de réponse (qui peut contenir "errors" —
        à inspecter par l'appelant), ou None si la clé est vide,
        si HTTP != 200 ou en cas d'erreur réseau/parse.
        Ne lève jamais d'exception.
        """

        if not self.api_key:
            print("      ⚠️ API-Football indisponible : clé API absente (.env)")
            return None

        self._request_count += 1

        try:
            response = requests.get(
                f"{self.BASE_URL}{endpoint}",
                headers={"x-apisports-key": self.api_key},
                params=params,
                timeout=15,
            )
        except requests.RequestException as e:
            print(f"      ⚠️ API-Football indisponible : {e}")
            return None

        if response.status_code != 200:
            print(f"      ⚠️ API-Football indisponible : "
                  f"HTTP {response.status_code}")
            return None

        try:
            return response.json()
        except ValueError:
            print("      ⚠️ API-Football indisponible : réponse illisible")
            return None

    @staticmethod
    def _errors_text(errors) -> str:
        """Aplatit le champ "errors" (dict ou liste) en texte lisible."""

        if isinstance(errors, dict):
            return " | ".join(f"{k}: {v}" for k, v in errors.items())
        if isinstance(errors, (list, tuple)):
            return " | ".join(str(e) for e in errors)
        return str(errors)

    # ─── RECHERCHE DU TEAM ID ───────────────────────

    # Mots génériques ignorés pour la recherche de repli
    _SEARCH_STOPWORDS = {
        "fc", "cf", "sc", "ac", "as", "ss", "ssc", "afc", "cfc",
        "club", "de", "du", "of", "the", "olympique", "losc", "ogc",
    }

    def _candidate_queries(self, team_name: str) -> List[str]:
        """
        Requêtes de recherche, de la plus précise à la plus large :
        nom complet → sans mots génériques → mot principal.
        (L'API appelle "Marseille" ce que Betclic nomme
        "Olympique de Marseille".)
        """

        full = self._normalize(team_name)
        queries = [full] if len(full) >= 3 else []

        words = [w for w in full.split() if w not in self._SEARCH_STOPWORDS]
        if words:
            reduced = " ".join(words)
            if len(reduced) >= 3 and reduced not in queries:
                queries.append(reduced)
            longest = max(words, key=len)
            if len(longest) >= 4 and longest not in queries:
                queries.append(longest)

        return queries

    def _search_team_id(self, team_name: str) -> Optional[int]:
        """
        Recherche l'ID API-Football d'une équipe par son nom.

        Privilégie une correspondance exacte (insensible à la
        casse/accents), puis une inclusion de nom, sinon le premier
        résultat de la requête la plus précise qui aboutit.
        """

        wanted = self._normalize(team_name)

        for query in self._candidate_queries(team_name):
            data = self._request("/teams", {"search": query})
            if not data:
                return None

            # Quota épuisé, paramètre invalide... → "errors" non vide
            errors = data.get("errors")
            if errors:
                print(f"      ⚠️ API-Football indisponible : "
                      f"{self._errors_text(errors)}")
                return None

            results = data.get("response") or []
            if not results:
                continue  # essayer la requête suivante

            # 1. Correspondance exacte
            for item in results:
                team = item.get("team") or {}
                if self._normalize(team.get("name", "")) == wanted:
                    return team.get("id")

            # 2. Inclusion (ex : "marseille" ⊂ "olympique de marseille")
            for item in results:
                team = item.get("team") or {}
                name = self._normalize(team.get("name", ""))
                if name and (name in wanted or wanted in name):
                    return team.get("id")

            # 3. Premier résultat de cette requête
            return (results[0].get("team") or {}).get("id")

        return None

    # ─── DERNIERS MATCHS TERMINÉS ───────────────────

    @staticmethod
    def _current_season() -> int:
        """Saison en cours au sens API-Football (2025 = saison 2025-26)."""

        now = datetime.now()
        return now.year if now.month >= 7 else now.year - 1

    @staticmethod
    def _max_allowed_season(errors) -> Optional[int]:
        """
        Extrait la saison max autorisée du message d'erreur du plan
        (ex : "Free plans do not have access ... try from 2022 to 2024.").
        """

        text = ApiFootballCollector._errors_text(errors)
        years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)]
        return max(years) if years else None

    def _remember_season(self, season: int):
        """Mémorise la saison accessible pour économiser les requêtes."""

        meta = self.cache.get("_meta")
        meta = meta if isinstance(meta, dict) else {}
        if meta.get("season") != season:
            meta["season"] = season
            self.cache["_meta"] = meta
            self._save_cache()

    def _fetch_last_fixtures(self, team_id: int) -> List[Dict]:
        """
        Récupère les derniers matchs terminés d'une équipe (max 15).

        Le plan gratuit refuse le paramètre `last` : on interroge la
        saison la plus récente autorisée (mémorisée dans le cache) et
        on filtre/trie côté client.
        """

        meta = self.cache.get("_meta")
        meta = meta if isinstance(meta, dict) else {}
        season = meta.get("season") or self._current_season()

        for _ in range(4):
            data = self._request("/fixtures", {
                "team": team_id,
                "season": season,
            })
            if not data:
                return []

            errors = data.get("errors")
            if errors:
                # Saison hors plan → redescendre sur la saison max autorisée
                allowed = self._max_allowed_season(errors)
                if allowed and allowed < season:
                    season = allowed
                    continue
                print(f"      ⚠️ API-Football indisponible : "
                      f"{self._errors_text(errors)}")
                return []

            fixtures = [
                fx for fx in (data.get("response") or [])
                if ((fx.get("fixture") or {}).get("status") or {})
                .get("short") in self.FINISHED_STATUSES
            ]
            if fixtures:
                self._remember_season(season)
                # Les plus récents d'abord, limités à 15
                fixtures.sort(
                    key=lambda fx: (fx.get("fixture") or {}).get("date") or "",
                    reverse=True,
                )
                return fixtures[:self.LAST_FIXTURES]

            # Saison vide (intersaison) → essayer la précédente
            season -= 1

        return []

    # ─── CALCUL DES STATS ───────────────────────────

    def _compute_stats(self, team_id: int,
                       fixtures: List[Dict]) -> Optional[Dict]:
        """
        Transforme les fixtures brutes en stats exploitables.

        Retourne None si moins de 5 matchs exploitables.
        """

        usable = []

        for fx in fixtures:
            goals = fx.get("goals") or {}
            teams = fx.get("teams") or {}
            gh, ga = goals.get("home"), goals.get("away")
            home_id = (teams.get("home") or {}).get("id")
            away_id = (teams.get("away") or {}).get("id")

            if gh is None or ga is None:
                continue
            if team_id not in (home_id, away_id):
                continue

            is_home = (home_id == team_id)
            usable.append({
                "date": (fx.get("fixture") or {}).get("date") or "",
                "is_home": is_home,
                "scored": gh if is_home else ga,
                "conceded": ga if is_home else gh,
            })

        if len(usable) < 5:
            return None

        # Les plus récents en premier (pour la forme)
        usable.sort(key=lambda m: m["date"], reverse=True)

        n = len(usable)
        avg_scored = sum(m["scored"] for m in usable) / n
        avg_conceded = sum(m["conceded"] for m in usable) / n

        home = [m for m in usable if m["is_home"]]
        away = [m for m in usable if not m["is_home"]]

        def _avg(matches: list, key: str, fallback: float) -> float:
            """Moyenne d'un champ, repli sur la moyenne globale si vide."""
            if not matches:
                return fallback
            return sum(m[key] for m in matches) / len(matches)

        # Forme récente : points/match sur les 5 plus récents
        points = 0
        for m in usable[:5]:
            if m["scored"] > m["conceded"]:
                points += 3
            elif m["scored"] == m["conceded"]:
                points += 1

        return {
            "avg_goals_scored": round(avg_scored, 2),
            "avg_goals_conceded": round(avg_conceded, 2),
            "avg_goals_scored_home": round(
                _avg(home, "scored", avg_scored), 2),
            "avg_goals_conceded_home": round(
                _avg(home, "conceded", avg_conceded), 2),
            "avg_goals_scored_away": round(
                _avg(away, "scored", avg_scored), 2),
            "avg_goals_conceded_away": round(
                _avg(away, "conceded", avg_conceded), 2),
            "recent_form_score": round(points / 5, 2),
            "matches_played": n,
            "last_match_date": (usable[0]["date"] or "")[:10],
        }

    # ─── FRAÎCHEUR DES DONNÉES ──────────────────────

    # Au-delà : les données sont trop vieilles pour être utiles
    MAX_DATA_AGE_DAYS = 550
    # Au-delà : moyennes de buts encore utiles, mais la "forme
    # récente" n'a plus de sens → neutralisée
    STALE_AFTER_DAYS = 60

    def _apply_freshness(self, stats: Dict) -> Optional[Dict]:
        """
        Le plan gratuit ne donne accès qu'aux saisons passées : il faut
        éviter de présenter des données périmées comme de la forme
        actuelle.
        """

        try:
            last = datetime.fromisoformat(stats.get("last_match_date", ""))
            age_days = (datetime.now() - last).days
        except (ValueError, TypeError):
            return stats  # date illisible : on garde tel quel

        if age_days > self.MAX_DATA_AGE_DAYS:
            return None

        if age_days > self.STALE_AFTER_DAYS:
            stats["recent_form_score"] = 1.5  # forme neutre
            stats["stale"] = True
            stats["age_days"] = age_days

        return stats

    # ─── POINT D'ENTRÉE ─────────────────────────────

    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """
        Retourne les stats réelles d'une équipe, ou None.

        Clés du dict retourné :
          avg_goals_scored, avg_goals_conceded,
          avg_goals_scored_home, avg_goals_conceded_home,
          avg_goals_scored_away, avg_goals_conceded_away,
          recent_form_score (points/match, 5 derniers),
          matches_played
        """

        if not team_name or not team_name.strip():
            return None

        key = self._normalize(team_name)
        entry = self.cache.get(key)
        entry = entry if isinstance(entry, dict) else {}

        # 1. Résultat frais en cache (y compris un échec) → zéro requête
        if "stats" in entry and (time.time() - entry.get("ts", 0)) < self.CACHE_TTL:
            return entry["stats"]

        # 2. team_id : cache (conservé à vie) ou recherche API
        team_id = entry.get("team_id")
        if not team_id:
            team_id = self._search_team_id(team_name)
            if not team_id:
                return None
            entry["team_id"] = team_id
            self.cache[key] = entry
            self._save_cache()  # le team_id est acquis, ne pas le reperdre

        # 3. Derniers matchs terminés → stats (avec contrôle de fraîcheur)
        fixtures = self._fetch_last_fixtures(team_id)
        stats = self._compute_stats(team_id, fixtures)
        if stats:
            stats = self._apply_freshness(stats)

        # Mémoriser même les échecs (stats=None) pour ne pas re-brûler
        # des requêtes à chaque analyse pendant 24h
        entry["ts"] = time.time()
        entry["stats"] = stats
        self.cache[key] = entry
        self._save_cache()

        if not stats:
            return None

        note = (f", saison précédente" if stats.get("stale") else "")
        print(f"      📡 API-Football : stats de {team_name} "
              f"({stats['matches_played']} matchs{note})")

        return stats
