"""
═══════════════════════════════════════════════════════
 MODULE CLOUD STORE — miroir permanent Supabase
═══════════════════════════════════════════════════════

Le SQLite de Streamlit Cloud s'efface à chaque redémarrage :
Supabase (PostgreSQL gratuit, API REST PostgREST) sert de
mémoire permanente ET de hub entre les appareils :

  • L'app (téléphone/cloud ou PC) pousse chaque analyse,
    règlement et CLV vers Supabase au fil de l'eau.
  • Au démarrage, hydrate_from_cloud() reconstitue le SQLite
    local depuis Supabase → toutes les pages existantes
    fonctionnent sans modification.
  • Le robot CLV du PC voit ainsi les paris faits au
    téléphone et leur capture aussi la cote de clôture.

Tables (créées via l'éditeur SQL Supabase, RLS désactivé —
l'accès est protégé par la clé API, elle-même chiffrée dans
le dépôt) :

  matches_cloud(match_key pk, updated_at, payload jsonb)
  bets_cloud(bet_key pk, match_key, result, closing_odds,
             clv_pct, updated_at, payload jsonb)

Toutes les méthodes sont silencieuses en cas d'échec réseau :
le cloud est un bonus, jamais un point de blocage.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from modules.secure_creds import get_supabase_creds

_TIMEOUT = 8
# Disjoncteur : après 3 échecs consécutifs, on cesse d'appeler
# Supabase pendant 5 min (évite 8 s de blocage par requête quand
# le projet gratuit est en pause ou le réseau coupé).
_BREAKER_THRESHOLD = 3
_BREAKER_PAUSE_S = 300


class CloudStore:
    """Client REST minimal pour le miroir Supabase."""

    def __init__(self, url: str, key: str):
        self.base = f"{url.rstrip('/')}/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self._failures = 0
        self._pause_until = 0.0

    # ─── REQUÊTE GÉNÉRIQUE (jamais d'exception) ──────

    def _req(self, method: str, table: str, params: Dict = None,
             body=None, prefer: str = None) -> Optional[requests.Response]:
        if time.time() < self._pause_until:
            return None

        headers = dict(self.headers)
        if prefer:
            headers["Prefer"] = prefer
        try:
            r = requests.request(
                method, f"{self.base}/{table}", headers=headers,
                params=params or {}, json=body, timeout=_TIMEOUT,
            )
            if r.status_code >= 400:
                print(f"   ⚠️ Supabase {method} {table}: "
                      f"HTTP {r.status_code}")
                self._register_failure()
                return None
            self._failures = 0
            return r
        except requests.RequestException as e:
            # type seulement : jamais l'URL/hostname dans les logs
            print(f"   ⚠️ Supabase injoignable ({method} {table}): "
                  f"{type(e).__name__}")
            self._register_failure()
            return None

    def _register_failure(self):
        self._failures += 1
        if self._failures >= _BREAKER_THRESHOLD:
            self._pause_until = time.time() + _BREAKER_PAUSE_S
            self._failures = 0
            print("   ⚠️ Supabase en pause 5 min (échecs répétés)")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ─── ÉCRITURES ───────────────────────────────────

    def upsert_match(self, match_key: str, payload: Dict) -> bool:
        return self._req(
            "POST", "matches_cloud",
            params={"on_conflict": "match_key"},
            body={"match_key": match_key, "payload": payload,
                  "updated_at": self._now()},
            prefer="resolution=merge-duplicates",
        ) is not None

    def upsert_bet(self, bet_key: str, match_key: str,
                   payload: Dict) -> bool:
        body = {
            "bet_key": bet_key,
            "match_key": match_key,
            "payload": payload,
            "updated_at": self._now(),
        }
        # Colonnes de tête omises quand None : un upsert de pari
        # « en attente » ne peut jamais effacer le résultat ou le
        # CLV d'une ligne déjà réglée (merge-duplicates ne met à
        # jour que les colonnes présentes dans le corps).
        for col in ("result", "closing_odds", "clv_pct"):
            if payload.get(col) is not None:
                body[col] = payload[col]

        return self._req(
            "POST", "bets_cloud",
            params={"on_conflict": "bet_key"},
            body=body,
            prefer="resolution=merge-duplicates",
        ) is not None

    def mark_superseded(self, home_team: str, away_team: str) -> bool:
        """
        Pierre tombale : marque « superseded » tous les paris EN
        ATTENTE de l'affiche dans le miroir, quelle que soit la
        date contenue dans la clé (couvre les analyses faites un
        autre jour ou sur un autre appareil). On ne SUPPRIME
        jamais : une ligne supprimée pourrait être ressuscitée par
        le push d'un appareil retardataire, la pierre tombale non.
        """
        return self._req(
            "PATCH", "bets_cloud",
            params={"match_key": f"like.{home_team}|{away_team}|*",
                    "result": "is.null"},
            body={"result": "superseded", "updated_at": self._now()},
        ) is not None

    def mark_superseded_key(self, bet_key: str) -> bool:
        """Pierre tombale ciblée (rattrapage d'un échec réseau)."""
        return self._req(
            "PATCH", "bets_cloud",
            params={"bet_key": f"eq.{bet_key}", "result": "is.null"},
            body={"result": "superseded", "updated_at": self._now()},
        ) is not None

    # ─── LECTURES ────────────────────────────────────

    def _fetch_all(self, table: str, pk: str) -> List[Dict]:
        # Pagination triée sur la clé primaire (unique et stable) :
        # aucun risque de ligne sautée/dupliquée entre deux pages.
        rows, page, step = [], 0, 1000
        while True:
            r = self._req("GET", table, params={
                "select": "*",
                "order": f"{pk}.asc",
                "limit": str(step),
                "offset": str(page * step),
            })
            if r is None:
                return rows
            chunk = r.json()
            rows.extend(chunk)
            if len(chunk) < step:
                return rows
            page += 1

    def fetch_matches(self) -> List[Dict]:
        return self._fetch_all("matches_cloud", "match_key")

    def fetch_bets(self) -> List[Dict]:
        return self._fetch_all("bets_cloud", "bet_key")

    def ping(self) -> bool:
        """Vérifie l'accès aux tables (setup / diagnostic)."""
        return self._req("GET", "matches_cloud",
                         params={"select": "match_key",
                                 "limit": "1"}) is not None


# ─── SINGLETON ───────────────────────────────────────

_instance: Optional[CloudStore] = None
_resolved = False


def get_cloud_store() -> Optional[CloudStore]:
    """CloudStore partagé, ou None si identifiants absents."""
    global _instance, _resolved
    if not _resolved:
        _resolved = True
        creds = get_supabase_creds()
        if creds:
            _instance = CloudStore(*creds)
    return _instance


def reset_cloud_store() -> None:
    """Force une nouvelle résolution (après saisie du code)."""
    global _instance, _resolved
    _instance, _resolved = None, False
