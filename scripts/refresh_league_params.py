"""
Mesure les paramètres réels de chaque championnat européen à partir
des résultats de football-data.co.uk, et les fige dans
data/league_params.json (committé → le cloud en profite sans réseau).

À relancer après chaque fin de saison (bouton « Actualiser les
données » du bureau).

Usage :
    python scripts/refresh_league_params.py
"""

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import config  # noqa: F401  (UTF-8 console + Paths)
from config import Paths, SUPPORTED_LEAGUES
from modules.football_data import FootballData

SORTIE = os.path.join(Paths.DATA_DIR, "league_params.json")


def main():
    fd = FootballData(offline=False)
    mesures = {}

    print("Mesure des paramètres réels par championnat\n")
    for cle in fd.DIVISIONS:
        params = fd.league_params(cle)
        if not params:
            print(f"  {cle:16s} — indisponible")
            continue

        defaut = SUPPORTED_LEAGUES.get(cle, {})
        mesures[cle] = params
        print(f"  {cle:16s} saison {params['saison']} "
              f"({params['matchs']:3d} matchs) : "
              f"{params['avg_goals']:.2f} buts | "
              f"{params['home_win_rate']*100:.1f}% dom | "
              f"1MT {params.get('first_half_share', 0)*100:.1f}%")

        for champ, libelle in (("avg_goals", "buts"),
                               ("home_win_rate", "dom"),
                               ("first_half_share", "1MT")):
            ancien, neuf = defaut.get(champ), params.get(champ)
            if isinstance(ancien, (int, float)) and neuf is not None:
                ecart = neuf - ancien
                if abs(ecart) >= 0.02:
                    print(f"       ↳ {libelle} : {ancien} → {neuf} "
                          f"({ecart:+.3f})")

    if not mesures:
        print("\nAucune mesure — rien n'est écrit.")
        return 1

    payload = {
        "source": "football-data.co.uk",
        "genere_par": "scripts/refresh_league_params.py",
        "ligues": mesures,
    }
    with io.open(SORTIE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"\n✅ {len(mesures)} championnats mesurés → {SORTIE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
