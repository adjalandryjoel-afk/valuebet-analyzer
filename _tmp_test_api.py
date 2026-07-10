"""Script temporaire — test live ApiFootballCollector (à supprimer)."""

from modules.api_football import ApiFootballCollector

print("═══ Appel 1 : API réelle attendue ═══")
c1 = ApiFootballCollector()
s1 = c1.get_team_stats("Paris Saint Germain")
print(f"Stats : {s1}")
print(f"Requêtes HTTP effectuées : {c1._request_count}")

print()
print("═══ Appel 2 : doit venir du cache (0 requête) ═══")
c2 = ApiFootballCollector()  # nouvelle instance → recharge le cache disque
s2 = c2.get_team_stats("Paris Saint Germain")
print(f"Stats : {s2}")
print(f"Requêtes HTTP effectuées : {c2._request_count}")

assert s1 is not None, "Appel 1 a échoué"
assert s2 == s1, "Le cache ne renvoie pas les mêmes stats"
assert c2._request_count == 0, "L'appel 2 a fait des requêtes HTTP"

print()
print("═══ Intégration DataCollector (0 requête, cache) ═══")
from modules.data_collector import DataCollector
dc = DataCollector()
ts = dc._stats_from_api("Paris Saint Germain")
print(f"TeamStats : {ts}")
assert ts is not None and ts.data_source == "api"
assert ts.points_per_game == ts.recent_form_score
assert dc.api_collector._request_count == 0

print()
print("✅ Test OK — API + cache + intégration opérationnels")
