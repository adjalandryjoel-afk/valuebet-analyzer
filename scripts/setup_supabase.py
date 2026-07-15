"""
Installation du miroir cloud Supabase.

Usage :
    python scripts/setup_supabase.py <PROJECT_URL> <ANON_KEY>

Chiffre les identifiants avec le code d'accès de l'app
(APP_ACCESS_CODE du .env) vers data/supabase_creds.enc
(commitable : le dépôt public ne voit que du chiffré),
puis vérifie l'accès réel aux tables.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import config  # noqa: F401  (charge .env + UTF-8 console)
from modules.secure_creds import encrypt_creds, get_supabase_creds
from modules.cloud_store import CloudStore, reset_cloud_store


def main():
    if len(sys.argv) != 3:
        print("Usage : python scripts/setup_supabase.py "
              "<PROJECT_URL> <ANON_KEY>")
        return 1

    url, key = sys.argv[1].strip(), sys.argv[2].strip()
    access_code = os.getenv("APP_ACCESS_CODE", "").strip()
    if not access_code:
        print("❌ APP_ACCESS_CODE absent du .env")
        return 1

    # 1. Test de connexion AVANT d'écrire quoi que ce soit
    store = CloudStore(url, key)
    if not store.ping():
        print("❌ Connexion refusée : URL/clé invalides ou tables "
              "absentes (exécuter scripts/supabase_schema.sql "
              "dans l'éditeur SQL Supabase d'abord).")
        return 1
    print("✅ Connexion Supabase OK, tables accessibles")

    # 2. Chiffrement et écriture
    encrypt_creds(url, key, access_code)
    print("✅ Identifiants chiffrés → data/supabase_creds.enc")

    # 3. Relecture complète du circuit (déchiffrement réel)
    reset_cloud_store()
    creds = get_supabase_creds()
    if not creds or creds[0] != url.rstrip("/"):
        print("❌ Relecture échouée — fichier chiffré incohérent")
        return 1
    if not CloudStore(*creds).ping():
        print("❌ Ping échoué après déchiffrement")
        return 1
    print("✅ Circuit complet vérifié (déchiffrement + accès)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
