"""
═══════════════════════════════════════════════════════
 MODULE SETTLEMENT — Règlement automatique des paris
 à partir du score final
═══════════════════════════════════════════════════════

Donne un score final (et optionnellement le score à la
mi-temps) → chaque pari en attente du match est réglé
(gagné/perdu), le profit calculé, le résultat du match
enregistré et l'Elo interne mis à jour.

Marchés décidables depuis le score : 1X2, Over/Under 2.5,
BTTS, totaux par équipe. Les mi-temps exigent le score HT ;
les tirs cadrés restent à régler manuellement.
"""

import re
from typing import Dict, Optional, Tuple

from modules.database_manager import DatabaseManager


# ══════════════════════════════════════════════════════
#  DÉCISION D'UNE SÉLECTION
# ══════════════════════════════════════════════════════

_LINE_RE = re.compile(r"(Over|Under)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def _over_under(selection: str, total: float) -> Optional[str]:
    """Décide un Over/Under X.5 contre un total de buts."""

    m = _LINE_RE.search(selection)
    if not m:
        return None
    direction, line = m.group(1).lower(), float(m.group(2))
    if direction == "over":
        return "win" if total > line else "loss"
    return "win" if total < line else "loss"


def settle_selection(market: str, selection: str,
                     home_team: str, away_team: str,
                     fthg: int, ftag: int,
                     hthg: Optional[int] = None,
                     htag: Optional[int] = None) -> Optional[str]:
    """
    Décide une sélection à partir du score.

    Retourne "win", "loss", ou None si indécidable (tirs cadrés,
    mi-temps sans score HT, marché inconnu).
    """

    market = (market or "").strip()
    selection = (selection or "").strip()

    # ── 1X2 ──
    if market == "1X2":
        reel = "1" if fthg > ftag else ("X" if fthg == ftag else "2")
        return "win" if selection == reel else "loss"

    # ── Buts du match ──
    if market == "Over/Under 2.5":
        return _over_under(selection, fthg + ftag)

    # ── BTTS ──
    if market == "BTTS":
        btts = fthg > 0 and ftag > 0
        if selection == "Oui":
            return "win" if btts else "loss"
        if selection == "Non":
            return "win" if not btts else "loss"
        return None

    # ── Tirs cadrés : indécidable depuis le score ──
    if market.startswith("Tirs cadrés"):
        return None

    # ── Buts par mi-temps (exige le score HT) ──
    if market == "Buts 1ère mi-temps":
        if hthg is None or htag is None:
            return None
        return _over_under(selection, hthg + htag)

    if market == "Buts 2ème mi-temps":
        if hthg is None or htag is None:
            return None
        return _over_under(selection, (fthg - hthg) + (ftag - htag))

    # ── Totaux par équipe : "Buts {équipe}" ──
    if market.startswith("Buts "):
        equipe = market[len("Buts "):].strip()
        if equipe == home_team:
            return _over_under(selection, fthg)
        if equipe == away_team:
            return _over_under(selection, ftag)
        return None

    return None


# ══════════════════════════════════════════════════════
#  RÈGLEMENT D'UN MATCH COMPLET
# ══════════════════════════════════════════════════════

def settle_match(db: DatabaseManager, match_id: int,
                 home_team: str, away_team: str,
                 fthg: int, ftag: int,
                 hthg: Optional[int] = None,
                 htag: Optional[int] = None,
                 elo=None) -> Dict:
    """
    Règle tous les paris en attente d'un match analysé.

    - marque chaque pari décidable gagné/perdu avec son profit
    - enregistre le score et le résultat réels du match
    - fait apprendre l'Elo interne (si un EloRatingSystem est fourni)

    Retourne {"regles", "gagnes", "perdus", "indecidables", "profit"}.
    """

    bilan = {"regles": 0, "gagnes": 0, "perdus": 0,
             "indecidables": 0, "profit": 0.0}

    pending = [b for b in db.get_pending_bets()
               if b.get("match_id") == match_id]

    for bet in pending:
        outcome = settle_selection(
            bet.get("market"), bet.get("selection"),
            home_team, away_team, fthg, ftag, hthg, htag,
        )

        if outcome is None:
            bilan["indecidables"] += 1
            continue

        stake = float(bet.get("recommended_stake") or 0)
        odds = float(bet.get("bookmaker_odds") or 0)
        profit = stake * (odds - 1) if outcome == "win" else -stake

        db.update_bet_result(bet["id"], outcome, profit)
        bilan["regles"] += 1
        bilan["profit"] += profit
        if outcome == "win":
            bilan["gagnes"] += 1
        else:
            bilan["perdus"] += 1

    # Résultat du match (alimente la calibration continue)
    resultat = "1" if fthg > ftag else ("X" if fthg == ftag else "2")
    db.update_match_result(match_id, f"{fthg}-{ftag}", resultat)

    # L'Elo interne apprend du résultat réel
    if elo is not None:
        try:
            elo.record_result(home_team, away_team, fthg, ftag)
            elo.save_ratings()
        except Exception:
            pass

    return bilan
