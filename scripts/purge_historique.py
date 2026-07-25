"""
Efface TOUT l'historique d'analyses de l'app : matchs, paris et
pierres tombales, en local ET dans le miroir cloud Supabase.

Une sauvegarde horodatée de la base est créée AVANT toute suppression
(data/backups/valuebet_AAAAMMJJ_HHMMSS.db) : la remise à zéro reste
récupérable en recopiant ce fichier sur data/valuebet.db.

Le cloud est purgé en même temps que le local — sinon
hydrate_from_cloud() restaurerait tout au prochain démarrage.

Usage :
    python scripts/purge_historique.py            # demande confirmation
    python scripts/purge_historique.py --oui      # sans confirmation
"""

import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import config  # noqa: F401,E402  (console UTF-8 + Paths)
from config import Paths  # noqa: E402
from modules.database_manager import DatabaseManager  # noqa: E402

DB_PATH = os.path.join(Paths.DATA_DIR, "valuebet.db")
BACKUP_DIR = os.path.join(Paths.DATA_DIR, "backups")


def sauvegarder() -> str:
    """Copie horodatée de la base. Retourne le chemin, ou ""."""
    if not os.path.exists(DB_PATH):
        return ""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BACKUP_DIR, f"valuebet_{stamp}.db")
    shutil.copy2(DB_PATH, dest)
    return dest


def main() -> int:
    db = DatabaseManager()

    # État avant
    stats = db.get_performance_stats()
    n_paris = stats.get("total_bets", 0) or 0
    print("Historique actuel :")
    print(f"  {n_paris} pari(s) enregistré(s)")
    print(f"  miroir cloud : "
          f"{'actif' if db.cloud else 'inactif (local seulement)'}")

    if "--oui" not in sys.argv:
        print("\nCette suppression est DÉFINITIVE (une sauvegarde sera "
              "créée).")
        rep = input("Confirmer la remise à zéro ? (oui/non) : ").strip()
        if rep.lower() not in ("oui", "o", "yes", "y"):
            print("Annulé — rien n'a été supprimé.")
            return 1

    backup = sauvegarder()
    if backup:
        print(f"\n💾 Sauvegarde : {backup}")

    res = db.purge_history(purge_cloud=True)

    print("\n✅ Historique effacé :")
    print(f"  matchs supprimés      : {res['matchs']}")
    print(f"  paris supprimés       : {res['paris']}")
    print(f"  pierres tombales      : {res['tombstones']}")
    if res["cloud"] is None:
        print("  cloud                 : inactif (rien à purger)")
    elif res["cloud"]:
        print("  cloud Supabase        : purgé ✓")
    else:
        print("  cloud Supabase        : ÉCHEC — relancer le script, "
              "sinon l'historique reviendra au prochain démarrage")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
