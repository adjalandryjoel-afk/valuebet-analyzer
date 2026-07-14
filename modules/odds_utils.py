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

import math
from typing import List, Optional

try:
    import shin as _shin           # wheel compilée (rapide), optionnelle
    HAS_SHIN = True
except ImportError:
    HAS_SHIN = False


def _shin_pure(odds: List[float]) -> Optional[List[float]]:
    """
    Méthode de Shin en pur Python (aucune dépendance compilée).

    Modèle de Shin (1992) : une part z des parieurs sont des
    « insiders ». Les probabilités s'obtiennent par
        p_i(z) = (√(z² + 4(1−z)·d_i²/β) − z) / (2(1−z))
    avec d_i = 1/cote_i et β = Σd_i. On résout Σp_i(z) = 1 par
    dichotomie sur z (la somme est décroissante en z, √β > 1 en z=0).
    """

    d = [1 / o for o in odds]
    beta = sum(d)
    if beta <= 1:  # pas de marge : cotes déjà justes
        return [x / beta for x in d]

    def prob_sum(z: float) -> float:
        return sum(
            (math.sqrt(z * z + 4 * (1 - z) * (di * di) / beta) - z)
            / (2 * (1 - z))
            for di in d
        )

    lo, hi = 0.0, 0.999
    if prob_sum(hi) > 1:  # marge extrême : insoluble proprement
        return None

    for _ in range(80):
        mid = (lo + hi) / 2
        if prob_sum(mid) > 1:
            lo = mid
        else:
            hi = mid

    z = (lo + hi) / 2
    probs = [
        (math.sqrt(z * z + 4 * (1 - z) * (di * di) / beta) - z)
        / (2 * (1 - z))
        for di in d
    ]
    total = sum(probs)
    return [p / total for p in probs]


def margin_ok(odds: List[float], max_margin: float = 0.15) -> bool:
    """
    Vérifie que la marge d'un marché est plausible (0 à max_margin).

    Une marge négative ou énorme signale des cotes corrompues
    (erreur de lecture OCR, mauvais marché) : elles ne doivent servir
    d'ancre à AUCUN calcul.
    """

    if not odds or any(o is None or o <= 1 for o in odds):
        return False
    margin = sum(1 / o for o in odds) - 1
    return 0 <= margin <= max_margin


def novig_probs(odds: List[float]) -> Optional[List[float]]:
    """
    Convertit une liste de cotes décimales (2 issues ou plus, toutes
    > 1) en probabilités sans marge via la méthode de Shin.

    Utilise la wheel compilée `shin` si disponible, sinon
    l'implémentation pure Python (identiques à ~1e-9), sinon la
    normalisation proportionnelle en dernier recours.

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

    # Implémentation pure Python (cloud : pas de wheel compilée)
    try:
        probs = _shin_pure(list(odds))
        if probs and all(0 < p < 1 for p in probs):
            return probs
    except Exception:
        pass

    # Dernier recours : normalisation proportionnelle
    inv = [1 / o for o in odds]
    total = sum(inv)
    return [v / total for v in inv]
