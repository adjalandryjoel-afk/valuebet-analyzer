"""
═══════════════════════════════════════════════════════
 MODULE API-FOOTBALL — Statistiques réelles des équipes
 via api-sports.io (accès direct, plan gratuit)
═══════════════════════════════════════════════════════

Fournit au modèle des statistiques INDÉPENDANTES des cotes :
buts marqués/encaissés (globaux et domicile/extérieur), forme
récente et 10 derniers résultats, calculés sur les 15 derniers
matchs terminés — plus le bilan des confrontations directes
(head-to-head) entre deux équipes.

Économie de requêtes (plan gratuit : 100 requêtes/jour) :
  • cache disque data/api_cache.json
  • team_id conservé indéfiniment (1 seule recherche/équipe)
  • stats conservées 24h (TTL), H2H conservés 7 jours
  • saison accessible mémorisée (le plan gratuit n'accepte ni
    le paramètre `last` ni les saisons trop récentes)

Toute erreur (clé absente, HTTP != 200, quota épuisé, réseau)
retourne None sans lever d'exception : le DataCollector
bascule alors sur ses sources de repli.
"""

import os
import re
import json
import math
import time
import unicodedata
import requests
from datetime import datetime
from typing import Dict, List, Optional

from config import APIKeys, Paths, PoissonConfig


# ══════════════════════════════════════════════════════
#  COLLECTEUR API-FOOTBALL
# ══════════════════════════════════════════════════════

_INSTANCE = None


def get_api_collector() -> "ApiFootballCollector":
    """
    Instance partagée du collecteur : un seul cache mémoire et un
    seul écrivain de data/api_cache.json pour toute l'application
    (DataCollector ET MatchIntelligence) — évite de doubler le quota
    et l'écrasement croisé du cache disque.
    """

    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ApiFootballCollector()
    return _INSTANCE


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

    # Durée de vie d'un bilan H2H en cache (secondes)
    H2H_CACHE_TTL = 7 * 24 * 3600

    # Nombre de derniers matchs terminés utilisés
    LAST_FIXTURES = 15

    # Nombre de confrontations directes conservées
    H2H_LAST_MATCHES = 10

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
            home = teams.get("home") or {}
            away = teams.get("away") or {}

            if gh is None or ga is None:
                continue
            if team_id not in (home.get("id"), away.get("id")):
                continue

            is_home = (home.get("id") == team_id)
            usable.append({
                "date": (fx.get("fixture") or {}).get("date") or "",
                "is_home": is_home,
                "scored": gh if is_home else ga,
                "conceded": ga if is_home else gh,
                "opponent": (away if is_home else home).get("name") or "",
            })

        if len(usable) < 5:
            return None

        # Les plus récents en premier (pour la forme)
        usable.sort(key=lambda m: m["date"], reverse=True)

        # Décote temporelle (Dixon-Coles time decay) : un match
        # d'il y a un an pèse ~2x moins qu'un match récent — crucial
        # avec le plan gratuit qui ne sert que des saisons passées.
        now = datetime.now()
        for m in usable:
            try:
                age = (now - datetime.fromisoformat(m["date"][:10])).days
            except (ValueError, TypeError):
                age = 365
            m["w"] = math.exp(-PoissonConfig.TIME_DECAY_XI * max(age, 0))

        # Poids total trop faible = données trop périmées pour signifier
        if sum(m["w"] for m in usable) < 2.5:
            return None

        n = len(usable)

        def _wavg(matches: list, key: str, fallback: float = None) -> float:
            """Moyenne pondérée par la fraîcheur des matchs."""
            total_w = sum(m["w"] for m in matches)
            if not matches or total_w <= 0:
                return fallback if fallback is not None else 0.0
            return sum(m[key] * m["w"] for m in matches) / total_w

        avg_scored = _wavg(usable, "scored")
        avg_conceded = _wavg(usable, "conceded")

        home = [m for m in usable if m["is_home"]]
        away = [m for m in usable if not m["is_home"]]

        def _avg(matches: list, key: str, fallback: float) -> float:
            """Moyenne pondérée d'un champ, repli si vide."""
            return _wavg(matches, key, fallback)

        # Forme récente : points/match sur les 5 plus récents
        points = 0
        for m in usable[:5]:
            if m["scored"] > m["conceded"]:
                points += 3
            elif m["scored"] == m["conceded"]:
                points += 1

        # 10 derniers résultats détaillés, du plus récent au plus ancien
        recent_results = [
            {
                "date": (m["date"] or "")[:10],
                "venue": "dom" if m["is_home"] else "ext",
                "adversaire": m["opponent"],
                "score": f"{m['scored']}-{m['conceded']}",
                "resultat": ("V" if m["scored"] > m["conceded"]
                             else "N" if m["scored"] == m["conceded"]
                             else "D"),
            }
            for m in usable[:10]
        ]

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
            "recent_results": recent_results,
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

    # ─── RÉSOLUTION D'UN TEAM ID (avec cache) ───────

    def _resolve_team_id(self, team_name: str) -> Optional[int]:
        """
        Retourne le team_id d'une équipe : cache disque (conservé à
        vie) en priorité, sinon recherche API mémorisée aussitôt.
        """

        key = self._normalize(team_name)
        entry = self.cache.get(key)
        entry = entry if isinstance(entry, dict) else {}

        team_id = entry.get("team_id")
        if team_id:
            return team_id

        team_id = self._search_team_id(team_name)
        if not team_id:
            return None

        entry["team_id"] = team_id
        self.cache[key] = entry
        self._save_cache()  # le team_id est acquis, ne pas le reperdre
        return team_id

    # ─── POINT D'ENTRÉE : STATS D'ÉQUIPE ────────────

    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """
        Retourne les stats réelles d'une équipe, ou None.

        Clés du dict retourné :
          avg_goals_scored, avg_goals_conceded,
          avg_goals_scored_home, avg_goals_conceded_home,
          avg_goals_scored_away, avg_goals_conceded_away,
          recent_form_score (points/match, 5 derniers),
          recent_results (10 derniers matchs, du plus récent au plus
            ancien : {date "YYYY-MM-DD", venue "dom"|"ext",
            adversaire, score "2-1" — buts de l'équipe d'abord,
            resultat "V"|"N"|"D"}),
          matches_played

        NB : une entrée mise en cache par une version antérieure peut
        ne pas contenir "recent_results" ; elle est retournée telle
        quelle (les consommateurs utilisent .get()).
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
        team_id = self._resolve_team_id(team_name)
        if not team_id:
            return None
        entry = self.cache.get(key)
        entry = entry if isinstance(entry, dict) else {}

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

    # ─── CONFRONTATIONS DIRECTES (H2H) ──────────────

    def _fetch_h2h_fixtures(self, id1: int, id2: int) -> Optional[List[Dict]]:
        """
        Récupère les confrontations directes terminées entre deux
        équipes (max 10, plus récentes d'abord).

        Le plan gratuit peut refuser la requête sans saison : on
        retente alors sur la saison max autorisée extraite du message
        d'erreur (même logique que _fetch_last_fixtures).

        Retourne None si aucune requête n'a abouti (clé absente,
        réseau, HTTP != 200) — à ne pas mettre en cache — et une
        liste (éventuellement vide) si l'API a répondu.
        """

        params = {"h2h": f"{id1}-{id2}"}
        data = self._request("/fixtures/headtohead", params)
        if data is None:
            return None

        errors = data.get("errors")
        if errors:
            # Saison/plan refusé → retenter sur la saison max autorisée
            allowed = self._max_allowed_season(errors)
            if not allowed:
                print(f"      ⚠️ API-Football indisponible : "
                      f"{self._errors_text(errors)}")
                return []
            data = self._request("/fixtures/headtohead",
                                 {**params, "season": allowed})
            if data is None:
                return None
            errors = data.get("errors")
            if errors:
                print(f"      ⚠️ API-Football indisponible : "
                      f"{self._errors_text(errors)}")
                return []
            self._remember_season(allowed)

        fixtures = [
            fx for fx in (data.get("response") or [])
            if ((fx.get("fixture") or {}).get("status") or {})
            .get("short") in self.FINISHED_STATUSES
        ]
        # Les plus récentes d'abord, limitées à 10
        fixtures.sort(
            key=lambda fx: (fx.get("fixture") or {}).get("date") or "",
            reverse=True,
        )
        return fixtures[:self.H2H_LAST_MATCHES]

    def _compute_h2h(self, team1_id: int, team2_id: int,
                     fixtures: List[Dict]) -> Optional[Dict]:
        """
        Transforme les confrontations brutes en bilan H2H relatif à
        team1. Retourne None si moins de 2 matchs exploitables.
        """

        matches = []
        team1_wins = draws = team2_wins = 0
        team1_goals = team2_goals = btts = 0

        for fx in fixtures:
            goals = fx.get("goals") or {}
            teams = fx.get("teams") or {}
            gh, ga = goals.get("home"), goals.get("away")
            home = teams.get("home") or {}
            away = teams.get("away") or {}

            if gh is None or ga is None:
                continue
            if {home.get("id"), away.get("id")} != {team1_id, team2_id}:
                continue

            team1_home = (home.get("id") == team1_id)
            scored1 = gh if team1_home else ga
            scored2 = ga if team1_home else gh

            matches.append({
                "date": (((fx.get("fixture") or {}).get("date")) or "")[:10],
                "home": home.get("name") or "",
                "away": away.get("name") or "",
                "score": f"{gh}-{ga}",
            })

            if scored1 > scored2:
                team1_wins += 1
            elif scored1 == scored2:
                draws += 1
            else:
                team2_wins += 1

            team1_goals += scored1
            team2_goals += scored2
            if scored1 > 0 and scored2 > 0:
                btts += 1

        n = len(matches)
        if n < 2:
            return None

        return {
            "matches": matches,
            "team1_wins": team1_wins,
            "draws": draws,
            "team2_wins": team2_wins,
            "avg_goals": round((team1_goals + team2_goals) / n, 2),
            "btts_rate": round(btts / n, 2),
            "team1_avg_scored": round(team1_goals / n, 2),
            "team2_avg_scored": round(team2_goals / n, 2),
            "sample": n,
        }

    @staticmethod
    def _orient_h2h(stats: Optional[Dict], flip: bool) -> Optional[Dict]:
        """
        Le bilan est mis en cache relatif à l'équipe au plus petit id
        (clé de cache triée) : si l'appelant a passé les équipes dans
        l'autre ordre, échanger les clés team1_*/team2_*.
        """

        if not stats or not flip:
            return stats

        oriented = dict(stats)
        oriented["team1_wins"] = stats["team2_wins"]
        oriented["team2_wins"] = stats["team1_wins"]
        oriented["team1_avg_scored"] = stats["team2_avg_scored"]
        oriented["team2_avg_scored"] = stats["team1_avg_scored"]
        return oriented

    # ─── POINT D'ENTRÉE : H2H ───────────────────────

    def get_h2h(self, team1_name: str, team2_name: str) -> Optional[Dict]:
        """
        Retourne le bilan des confrontations directes (10 dernières,
        matchs terminés uniquement), ou None si moins de 2
        confrontations trouvées ou en cas d'erreur. Ne lève jamais
        d'exception.

        Clés du dict retourné (team1 = premier argument) :
          matches (du plus récent au plus ancien :
            {date "YYYY-MM-DD", home, away, score "2-1" — buts de
            l'équipe à domicile d'abord}),
          team1_wins, draws, team2_wins,
          avg_goals (buts totaux/match), btts_rate (part des matchs
          où les deux marquent), team1_avg_scored, team2_avg_scored,
          sample (nb de matchs)

        Cache disque 7 jours sous "h2h::{id_min}-{id_max}", échecs
        compris (stats: None) pour ne pas re-brûler du quota.
        """

        if not team1_name or not team1_name.strip():
            return None
        if not team2_name or not team2_name.strip():
            return None

        # 1. Résolution des deux team_id (cache à vie ou recherche)
        id1 = self._resolve_team_id(team1_name)
        id2 = self._resolve_team_id(team2_name)
        if not id1 or not id2 or id1 == id2:
            return None

        low, high = sorted((id1, id2))
        flip = (id1 != low)  # bilan stocké relatif au plus petit id
        cache_key = f"h2h::{low}-{high}"

        # 2. Bilan frais en cache (y compris un échec) → zéro requête
        entry = self.cache.get(cache_key)
        entry = entry if isinstance(entry, dict) else {}
        if "stats" in entry and \
                (time.time() - entry.get("ts", 0)) < self.H2H_CACHE_TTL:
            return self._orient_h2h(entry["stats"], flip)

        # 3. Confrontations terminées → bilan
        fixtures = self._fetch_h2h_fixtures(low, high)
        if fixtures is None:
            return None  # requête impossible : ne pas figer 7 jours

        stats = self._compute_h2h(low, high, fixtures)

        # Mémoriser même les échecs (stats=None) pour ne pas
        # re-brûler du quota à chaque analyse pendant 7 jours
        entry["ts"] = time.time()
        entry["stats"] = stats
        self.cache[cache_key] = entry
        self._save_cache()

        if not stats:
            return None

        print(f"      📡 API-Football : H2H {team1_name} / {team2_name} "
              f"({stats['sample']} confrontations)")

        return self._orient_h2h(stats, flip)
