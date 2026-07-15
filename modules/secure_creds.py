"""
═══════════════════════════════════════════════════════
 MODULE SECURE CREDS — identifiants Supabase chiffrés
═══════════════════════════════════════════════════════

Le dépôt GitHub est public : les identifiants Supabase n'y
sont JAMAIS commités en clair. Ils sont chiffrés (Fernet /
AES-128-CBC + HMAC) avec une clé dérivée du code d'accès de
l'app (PBKDF2-SHA256, 300 000 itérations) — le même code que
le verrou d'entrée, déjà connu de l'utilisateur et jamais
stocké dans le dépôt.

  • Sur le cloud : le code saisi au verrou est placé dans
    l'environnement (APP_ACCESS_CODE) et sert au déchiffrement.
  • En local : APP_ACCESS_CODE vient du fichier .env.
  • Les variables SUPABASE_URL / SUPABASE_KEY, si présentes,
    court-circuitent le fichier chiffré (mode développeur).
"""

import base64
import hashlib
import io
import json
import os
from typing import Optional, Tuple

from config import Paths

CREDS_PATH = os.path.join(Paths.DATA_DIR, "supabase_creds.enc")
PBKDF2_ITERATIONS = 300_000


def _derive_key(access_code: str, salt: bytes) -> bytes:
    """Clé Fernet (32 octets base64) dérivée du code d'accès."""
    raw = hashlib.pbkdf2_hmac(
        "sha256", access_code.encode("utf-8"), salt,
        PBKDF2_ITERATIONS, dklen=32,
    )
    return base64.urlsafe_b64encode(raw)


def encrypt_creds(url: str, key: str, access_code: str,
                  path: str = CREDS_PATH) -> None:
    """Chiffre et écrit les identifiants Supabase (commitable)."""
    from cryptography.fernet import Fernet

    salt = os.urandom(16)
    fernet = Fernet(_derive_key(access_code.strip(), salt))
    token = fernet.encrypt(json.dumps(
        {"url": url.strip().rstrip("/"), "key": key.strip()}
    ).encode("utf-8"))

    payload = {
        "algo": f"fernet+pbkdf2_sha256_{PBKDF2_ITERATIONS}",
        "salt": salt.hex(),
        "token": token.decode("ascii"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def get_supabase_creds(path: str = CREDS_PATH
                       ) -> Optional[Tuple[str, str]]:
    """
    (url, key) Supabase, ou None si indisponible.
    Ordre : variables d'environnement, puis fichier chiffré.
    Ne lève jamais : l'app fonctionne sans cloud.
    """

    env_url = os.getenv("SUPABASE_URL", "").strip()
    env_key = os.getenv("SUPABASE_KEY", "").strip()
    if env_url and env_key:
        return env_url.rstrip("/"), env_key

    access_code = os.getenv("APP_ACCESS_CODE", "").strip()
    if not access_code or not os.path.exists(path):
        return None

    try:
        from cryptography.fernet import Fernet, InvalidToken

        with io.open(path, encoding="utf-8") as f:
            payload = json.load(f)

        fernet = Fernet(_derive_key(
            access_code, bytes.fromhex(payload["salt"])))
        try:
            data = json.loads(
                fernet.decrypt(payload["token"].encode("ascii")))
        except InvalidToken:
            return None  # mauvais code d'accès

        url, key = data.get("url", ""), data.get("key", "")
        return (url.rstrip("/"), key) if url and key else None
    except Exception:
        return None
