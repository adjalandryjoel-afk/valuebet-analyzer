"""Sonde temporaire — quelles requêtes fixtures le plan gratuit accepte (à supprimer)."""

import requests
from config import APIKeys

H = {"x-apisports-key": APIKeys.RAPIDAPI_KEY}
B = "https://v3.football.api-sports.io"

for season in (2025, 2023):
    r = requests.get(f"{B}/fixtures", headers=H,
                     params={"team": 85, "season": season}, timeout=15)
    j = r.json()
    print(f"season={season} http={r.status_code} errors={j.get('errors')} "
          f"results={j.get('results')}")
    resp = j.get("response") or []
    if resp:
        dates = sorted(fx["fixture"]["date"] for fx in resp)
        statuses = {fx["fixture"]["status"]["short"] for fx in resp}
        print(f"   plage dates : {dates[0][:10]} → {dates[-1][:10]}")
        print(f"   statuts : {sorted(statuses)}")
        break
