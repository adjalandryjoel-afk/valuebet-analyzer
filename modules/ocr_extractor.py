"""
═══════════════════════════════════════════════════════
 MODULE OCR EXTRACTOR — Extraction des matchs et cotes
 depuis les captures d'écran Betclic
═══════════════════════════════════════════════════════

Deux moteurs, par ordre de priorité :
1. OpenAI Vision (gpt-4o-mini) — le plus fiable, nécessite
   OPENAI_API_KEY dans le .env
2. Tesseract OCR local (pytesseract) — fallback gratuit,
   nécessite Tesseract installé sur la machine

Format de sortie standard par match :
{
    "equipe_domicile": str,
    "equipe_exterieur": str,
    "competition": str,
    "cotes": {"1": float, "X": float, "2": float},
    "marches_supplementaires": {"over_2_5": ..., "btts_oui": ...}
}
"""

import os
import re
import json
import base64
from typing import Dict, List, Optional

from config import APIKeys, OCRConfig


# ══════════════════════════════════════════════════════
#  PROMPT D'EXTRACTION VISION
# ══════════════════════════════════════════════════════

VISION_PROMPT = """Tu analyses une capture d'écran de l'application \
Betclic Côte d'Ivoire (paris sportifs football).

Extrais TOUS les matchs visibles avec leurs cotes. Réponds UNIQUEMENT \
avec un JSON valide, sans texte autour, au format :

{
  "matchs": [
    {
      "equipe_domicile": "nom exact affiché",
      "equipe_exterieur": "nom exact affiché",
      "competition": "compétition si visible, sinon \\"\\"",
      "cotes": {"1": 1.85, "X": 3.40, "2": 4.20},
      "marches_supplementaires": {
        "over_2_5": 1.72,
        "under_2_5": 2.10,
        "btts_oui": 1.85,
        "btts_non": 1.95,
        "home_over_1_5": 1.60,
        "away_under_0_5": 3.25,
        "h1_over_0_5": 1.28,
        "h2_under_1_5": 1.55,
        "sot_home_over_3_5": 1.80,
        "sot_away_under_4_5": 1.90
      }
    }
  ]
}

Règles STRICTES :
- Les cotes sont des nombres décimaux (point, pas virgule)
- "cotes" (1/X/2) = UNIQUEMENT le marché explicitement libellé \
"Résultat du match", "1N2", "1X2" ou "Vainqueur" avec ses 3 issues. \
Ne JAMAIS y mettre les cotes d'un autre marché (double chance, \
handicap, mi-temps, buteur...). Si ce marché n'est pas visible, \
mets {"1": 0, "X": 0, "2": 0}
- "over_2_5"/"under_2_5" = UNIQUEMENT la ligne exactement 2,5 buts \
du marché "Nombre de buts" du MATCH ENTIER. Ignore les lignes 0,5 / \
1,5 / 3,5 / 4,5 etc.
- "btts_oui"/"btts_non" = uniquement le marché "Les deux équipes \
marquent"
- Totaux par équipe (marché "Nombre de buts de [équipe]" ou "Total \
de buts équipe") : clés home_over_0_5, home_under_0_5, home_over_1_5, \
home_under_1_5, home_over_2_5, home_under_2_5 (idem away_). home_ = \
la 1ère équipe affichée (domicile), away_ = la 2ème (extérieur). \
Lignes 0,5 / 1,5 / 2,5 uniquement. Ne JAMAIS confondre le total \
d'une équipe avec le total du match
- Buts par mi-temps du match ENTIER (marché "Nombre de buts 1ère \
mi-temps" / "2ème mi-temps", PAS par équipe) : clés h1_over_0_5, \
h1_under_0_5, h1_over_1_5, h1_under_1_5 (idem h2_). Lignes 0,5 / \
1,5 uniquement
- Tirs cadrés par équipe (marché "Tirs cadrés de [équipe]") : clés \
sot_home_over_N_5 / sot_home_under_N_5 (idem sot_away_) avec N,5 = \
la ligne réellement affichée (ex. plus de 3,5 tirs cadrés domicile \
→ sot_home_over_3_5). Ne JAMAIS confondre tirs cadrés et buts
- N'inclus dans marches_supplementaires que les marchés réellement \
visibles et identifiés avec certitude — en cas de doute sur le \
marché ou la ligne, omets la clé
- Une capture peut montrer plusieurs écrans de marchés du MÊME match : \
c'est un seul match, pas plusieurs
- Si l'image n'est pas une capture de bookmaker (appli de stats, \
scores...), réponds {"matchs": []}"""


# ══════════════════════════════════════════════════════
#  FUSION DES CAPTURES D'UN MÊME MATCH
# ══════════════════════════════════════════════════════

def merge_matches(matches: List[Dict]) -> List[Dict]:
    """
    Fusionne les matchs extraits de plusieurs captures.

    L'utilisateur capture souvent plusieurs écrans de marchés du même
    match (1X2, buts, BTTS...). Sans fusion, chaque capture devient un
    « match » séparé avec des cotes incomplètes — et les marchés
    orphelins produisent de fausses values.
    """

    def norm(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    merged: Dict[tuple, Dict] = {}
    order = []

    for m in matches:
        key = (norm(m.get("equipe_domicile", "")),
               norm(m.get("equipe_exterieur", "")))

        if key not in merged:
            merged[key] = {
                "equipe_domicile": m.get("equipe_domicile", ""),
                "equipe_exterieur": m.get("equipe_exterieur", ""),
                "competition": m.get("competition", ""),
                "cotes": {},
                "marches_supplementaires": {},
            }
            order.append(key)

        target = merged[key]

        if not target["competition"] and m.get("competition"):
            target["competition"] = m["competition"]

        # Première cote non nulle rencontrée = référence
        for k, v in (m.get("cotes") or {}).items():
            try:
                v = float(v or 0)
            except (TypeError, ValueError):
                continue
            if v > 1 and float(target["cotes"].get(k, 0) or 0) <= 1:
                target["cotes"][k] = v

        for k, v in (m.get("marches_supplementaires") or {}).items():
            try:
                v = float(v or 0)
            except (TypeError, ValueError):
                continue
            if v > 1 and float(
                target["marches_supplementaires"].get(k, 0) or 0
            ) <= 1:
                target["marches_supplementaires"][k] = v

    return [merged[k] for k in order]


# ══════════════════════════════════════════════════════
#  EXTRACTEUR
# ══════════════════════════════════════════════════════

class BetclicScreenshotExtractor:
    """Extrait matchs et cotes des captures d'écran Betclic."""

    def __init__(self):
        self.openai_client = None

        if APIKeys.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=APIKeys.OPENAI_API_KEY)
            except ImportError:
                print("    ⚠️ Package openai non installé — vision désactivée")

        self.has_tesseract = self._check_tesseract()

        if not self.openai_client and not self.has_tesseract:
            print("    ⚠️ Aucun moteur OCR disponible "
                  "(ni OPENAI_API_KEY, ni Tesseract)")

    @staticmethod
    def _check_tesseract() -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    # ─── EXTRACTION D'UNE IMAGE ─────────────────────

    def extract_from_image(self, image_path: str) -> Dict:
        """
        Extrait les matchs d'une capture d'écran.

        Retourne {"matchs": [...], "source": ..., "error": ...}
        """

        if not os.path.exists(image_path):
            return {"matchs": [], "error": f"Fichier introuvable : {image_path}"}

        # 1. OpenAI Vision (prioritaire)
        if self.openai_client:
            result = self._extract_with_vision(image_path)
            if result is not None:
                return result

        # 2. Tesseract (fallback)
        if self.has_tesseract:
            return self._extract_with_tesseract(image_path)

        return {
            "matchs": [],
            "error": "Aucun moteur OCR disponible. Ajoutez OPENAI_API_KEY "
                     "au .env ou installez Tesseract, ou utilisez la saisie "
                     "manuelle dans l'interface Streamlit."
        }

    # ─── MOTEUR 1 : OPENAI VISION ───────────────────

    def _extract_with_vision(self, image_path: str) -> Optional[Dict]:
        """Extraction via l'API vision d'OpenAI."""

        try:
            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')

            ext = os.path.splitext(image_path)[1].lstrip('.').lower()
            if ext == "jpg":
                ext = "jpeg"

            response = self.openai_client.chat.completions.create(
                model=OCRConfig.VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{image_b64}"
                            }
                        },
                    ],
                }],
                max_tokens=2000,
                temperature=0,
            )

            content = response.choices[0].message.content.strip()

            # Nettoyer les éventuels blocs markdown
            content = re.sub(r"^```(json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

            data = json.loads(content)
            data["source"] = "openai_vision"
            return data

        except json.JSONDecodeError:
            print(f"      ⚠️ Vision : réponse JSON invalide pour {image_path}")
            return {"matchs": [], "source": "openai_vision",
                    "error": "JSON invalide"}
        except Exception as e:
            print(f"      ⚠️ Vision indisponible ({e}) — fallback OCR local")
            return None

    # ─── MOTEUR 2 : TESSERACT ───────────────────────

    def _extract_with_tesseract(self, image_path: str) -> Dict:
        """
        Extraction basique via Tesseract : cherche des lignes
        "Équipe A - Équipe B" suivies de trois cotes décimales.
        """

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='fra+eng')

        except Exception as e:
            return {"matchs": [], "source": "tesseract",
                    "error": f"OCR échoué : {e}"}

        matches = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        odds_pattern = re.compile(r"\b(\d{1,2}[.,]\d{2})\b")
        vs_pattern = re.compile(
            r"^(.{3,30}?)\s*[-–—]\s*(.{3,30})$"
        )

        pending_teams = None

        for line in lines:
            odds_found = odds_pattern.findall(line)

            team_match = vs_pattern.match(line)
            if team_match and len(odds_found) < 2:
                pending_teams = (
                    team_match.group(1).strip(),
                    team_match.group(2).strip(),
                )
                continue

            if pending_teams and len(odds_found) >= 3:
                odds = [float(o.replace(',', '.')) for o in odds_found[:3]]
                # Filtrer les faux positifs (scores, heures...)
                if all(1.01 <= o <= 50 for o in odds):
                    matches.append({
                        "equipe_domicile": pending_teams[0],
                        "equipe_exterieur": pending_teams[1],
                        "competition": "",
                        "cotes": {"1": odds[0], "X": odds[1], "2": odds[2]},
                        "marches_supplementaires": {},
                    })
                pending_teams = None

        return {"matchs": matches, "source": "tesseract"}

    # ─── EXTRACTION D'UN DOSSIER ────────────────────

    def extract_from_directory(self, directory: str) -> List[Dict]:
        """
        Extrait les matchs de toutes les images d'un dossier.

        Retourne directement la liste des matchs (tous fichiers confondus).
        """

        if not os.path.isdir(directory):
            print(f"  ⚠️ Dossier introuvable : {directory}")
            return []

        image_files = sorted(
            f for f in os.listdir(directory)
            if f.lower().endswith(OCRConfig.ALLOWED_EXTENSIONS)
        )[:OCRConfig.MAX_IMAGES]

        if not image_files:
            print(f"  ⚠️ Aucune image dans {directory}")
            return []

        print(f"\n  📸 {len(image_files)} capture(s) à analyser...")

        all_matches = []

        for filename in image_files:
            path = os.path.join(directory, filename)
            print(f"    🔍 {filename}...")

            result = self.extract_from_image(path)
            found = result.get("matchs", [])

            if found:
                print(f"       ✅ {len(found)} match(s) extraits")
                all_matches.extend(found)
            else:
                err = result.get("error", "aucun match détecté")
                print(f"       ❌ {err}")

        merged = merge_matches(all_matches)
        if len(merged) < len(all_matches):
            print(f"  🔗 {len(all_matches)} extraction(s) fusionnées en "
                  f"{len(merged)} match(s) distinct(s)")

        return merged


def create_extractor() -> BetclicScreenshotExtractor:
    """Factory : crée l'extracteur configuré."""
    return BetclicScreenshotExtractor()
