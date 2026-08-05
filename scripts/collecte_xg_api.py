"""
═══════════════════════════════════════════════════════════════
 COLLECTE xG API-FOOTBALL — championnats secondaires
═══════════════════════════════════════════════════════════════

Constitue l'historique xG match par match des championnats que
Understat ne couvre pas, pour permettre au backtest de mesurer si
le xG apporte quelque chose LÀ OÙ LES MARCHÉS SONT MOINS EFFICIENTS.

Understat s'arrête aux 5 grands championnats et FBref refuse les
requêtes directes (HTTP 403) ; l'abonnement API-Football est donc
la seule source historique disponible. Couverture mesurée avant
lancement : Championship, Primeira Liga, Jupiler Pro et Süper Lig
à 100 % sur trois saisons ; Eredivisie à 88 % ; Grèce, Écosse et
2. Bundesliga sans aucun xG avant 2025.

COÛT : une requête par match (aucun lot possible, vérifié :
« The Fixture field must contain an integer »). Environ 4 550
requêtes pour 4 championnats × 3 saisons.

REPRISE : le cache est écrit au fil de l'eau. Une interruption ne
fait rien perdre et le script reprend là où il s'était arrêté.

Usage :  python -X utf8 scripts/collecte_xg_api.py
"""

import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import APIKeys, Paths                       # noqa: E402

BASE = "https://v3.football.api-sports.io"
CACHE = os.path.join(Paths.DATA_DIR, "football_data",
                     "xg_api_secondaires.json")

# (id API-Football, division football-data, nom lisible)
LIGUES = [
    (40, "E1", "Championship"),
    (94, "P1", "Primeira Liga"),
    (144, "B1", "Jupiler Pro League"),
    (203, "T1", "Süper Lig"),
]

# saison API-Football → saison football-data
SAISONS = {2023: "2324", 2024: "2425", 2025: "2526"}

STATUTS_JOUES = ("FT", "AET", "PEN")

# Débit : le plan Pro annonce 450 req/min mais ce chiffre n'est pas
# vérifiable via /status. On reste nettement en dessous et on ralentit
# au premier 429 — c'est la réponse de l'API qui décide, pas la doc.
WORKERS = 6
INTERVALLE_MIN = 0.10

_verrou = threading.Lock()
_dernier = [0.0]
_intervalle = [INTERVALLE_MIN]
_n_429 = [0]


def _appel(ep, params):
    """Requête régulée, avec ralentissement automatique sur 429."""

    for essai in range(4):
        with _verrou:
            attente = _intervalle[0] - (time.time() - _dernier[0])
            if attente > 0:
                time.sleep(attente)
            _dernier[0] = time.time()
        try:
            r = requests.get(f"{BASE}{ep}",
                             headers={"x-apisports-key":
                                      APIKeys.RAPIDAPI_KEY},
                             params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 + essai * 2)
            continue

        if r.status_code == 429:
            with _verrou:
                _n_429[0] += 1
                _intervalle[0] = min(2.0, _intervalle[0] * 2)
            time.sleep(3 + essai * 3)
            continue
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None
    return None


def _xg_du_match(fixture_id):
    """(xg_domicile, xg_extérieur) ou None."""

    data = _appel("/fixtures/statistics", {"fixture": fixture_id})
    if not data or data.get("errors"):
        return None
    reponse = data.get("response") or []
    if len(reponse) < 2:
        return None

    valeurs = []
    for equipe in reponse[:2]:
        stats = {s.get("type"): s.get("value")
                 for s in (equipe.get("statistics") or [])}
        brut = stats.get("expected_goals")
        try:
            valeurs.append(float(brut))
        except (TypeError, ValueError):
            return None          # une seule équipe sans xG → inutilisable
    return tuple(valeurs)


def _charger_cache():
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _sauver_cache(cache):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE)


def main():
    cache = _charger_cache()
    deja = sum(len(v) for v in cache.values())
    print(f"Cache existant : {deja} matchs")

    total_appels = 0
    t0 = time.time()

    for lid, div, nom in LIGUES:
        for saison_api, saison_fd in SAISONS.items():
            cle_bloc = f"{saison_fd}|{div}"
            bloc = cache.setdefault(cle_bloc, {})

            data = _appel("/fixtures", {"league": lid,
                                        "season": saison_api})
            total_appels += 1
            fixtures = [
                f for f in ((data or {}).get("response") or [])
                if ((f.get("fixture") or {}).get("status") or {})
                .get("short") in STATUTS_JOUES
            ]
            # Le nom d'équipe et la date servent de clé de jointure
            # avec football-data ; l'appariement lui-même est fait
            # côté backtest, par (date, score), jamais par les noms.
            a_faire = []
            for f in fixtures:
                fid = str(f["fixture"]["id"])
                if fid in bloc:
                    continue
                a_faire.append(f)

            if not a_faire:
                print(f"  {nom:20s} {saison_fd}  déjà complet "
                      f"({len(bloc)} matchs)")
                continue

            print(f"  {nom:20s} {saison_fd}  {len(a_faire)} matchs "
                  f"à collecter ...", flush=True)

            fait = [0]

            def traiter(f):
                fid = f["fixture"]["id"]
                xg = _xg_du_match(fid)
                eq = f.get("teams") or {}
                but = f.get("goals") or {}
                res = {
                    "date": (f["fixture"]["date"] or "")[:10],
                    "home": (eq.get("home") or {}).get("name"),
                    "away": (eq.get("away") or {}).get("name"),
                    "gh": but.get("home"),
                    "ga": but.get("away"),
                    "xg": list(xg) if xg else None,
                }
                fait[0] += 1
                if fait[0] % 100 == 0:
                    print(f"      {fait[0]}/{len(a_faire)} "
                          f"({time.time()-t0:.0f}s, "
                          f"{_n_429[0]} ralentissements)", flush=True)
                return str(fid), res

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for fid, res in pool.map(traiter, a_faire):
                    bloc[fid] = res
            total_appels += len(a_faire)

            avec = sum(1 for v in bloc.values() if v.get("xg"))
            print(f"      → {len(bloc)} matchs, {avec} avec xG "
                  f"({avec/max(len(bloc),1):.0%})", flush=True)
            _sauver_cache(cache)

    _sauver_cache(cache)
    total = sum(len(v) for v in cache.values())
    avec = sum(1 for v in cache.values() for m in v.values()
               if m.get("xg"))
    print()
    print(f"TERMINÉ — {total} matchs, {avec} avec xG "
          f"({avec/max(total,1):.0%})")
    print(f"{total_appels} requêtes, {_n_429[0]} ralentissements, "
          f"{time.time()-t0:.0f}s")
    print(f"Cache → {CACHE}")


if __name__ == "__main__":
    main()
