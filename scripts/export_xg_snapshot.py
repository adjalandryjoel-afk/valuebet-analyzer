"""
═══════════════════════════════════════════════════════
 EXPORT DU SNAPSHOT xG — à lancer depuis le PC
═══════════════════════════════════════════════════════

Calcule les profils xG de TOUTES les équipes des 5 grands
championnats (via soccerdata/Understat, installé en local
uniquement) et les publie dans data/xg_snapshot.json.

Ce fichier est committé sur GitHub : l'app cloud (qui n'a
pas soccerdata) le lit en repli — le téléphone profite du
xG sans dépendance lourde.

Usage :  python scripts/export_xg_snapshot.py
Ensuite : git add data/xg_snapshot.json && git commit && git push
(ou double-clic sur « Actualiser les données.bat »)
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.xg_provider import XgProvider  # noqa: E402


def main():
    provider = XgProvider()
    profils = {}

    for league, sd_league in XgProvider.LEAGUE_MAP.items():
        print(f"\n📊 {league} ({sd_league})...")

        exporte = 0
        for season in provider.SEASONS:
            df = provider._load_league(sd_league, season)
            if df is None or getattr(df, "empty", True):
                continue

            teams = sorted(set(df["home_team"]) | set(df["away_team"]))
            for team in teams:
                key = f"{provider._normalize(team)}|{league}"
                if key in profils:
                    continue
                try:
                    profile = provider._profile_from_matches(
                        df, team, season
                    )
                except Exception:
                    profile = None
                if profile:
                    profils[key] = profile
                    exporte += 1

            if exporte:
                break  # la saison la plus récente suffit

        print(f"   → {exporte} équipe(s) exportée(s)")

    snapshot = {
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "n_profils": len(profils),
        "profils": profils,
    }

    out = XgProvider.SNAPSHOT_PATH
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1)

    print(f"\n✅ Snapshot écrit : {out} ({len(profils)} profils)")


if __name__ == "__main__":
    main()
