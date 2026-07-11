"""
═══════════════════════════════════════════════════════
 MODULE ODDS UTILS — Conversion cotes → probabilités
 sans marge (méthode de Shin)
═══════════════════════════════════════════════════════

La marge du bookmaker n'est pas répartie également entre les
issues : elle charge davantage les outsiders (biais
favori-outsider). La normalisation proportionnelle (1/cote
divisé par la somme) sous-estime donc les favoris et surestime
les outsiders.

La méthode de Shin (1992-1993) modélise ce biais et donne les
probabilités les plus précises dans la littérature comparative
(Štrumbelj 2014). On l'utilise partout où l'app retire la marge ;
repli sur la normalisation proportionnelle si le calcul échoue.
"""

from typing import List, Optional

try:
    import shin as _shin
    HAS_SHIN = True
except ImportError:
    HAS_SHIN = False


def novig_probs(odds: List[float]) -> Optional[List[float]]:
    """
    Convertit une liste de cotes décimales (2 issues ou plus, toutes
    > 1) en probabilités sans marge via la méthode de Shin.

    Retourne None si les cotes sont inexploitables.
    """

    if not odds or any(o is None or o <= 1 for o in odds):
        return None

    if HAS_SHIN:
        try:
            probs = _shin.calculate_implied_probabilities(list(odds))
            # Le package peut retourner un dict selon la version
            if isinstance(probs, dict):
                probs = probs.get("implied_probabilities", probs)
            probs = list(probs)
            if (len(probs) == len(odds)
                    and all(0 < p < 1 for p in probs)
                    and abs(sum(probs) - 1) < 1e-6):
                return probs
        except Exception:
            pass

    # Repli : normalisation proportionnelle
    inv = [1 / o for o in odds]
    total = sum(inv)
    return [v / total for v in inv]
