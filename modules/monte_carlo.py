"""
═══════════════════════════════════════════════════════
 MODULE MONTE CARLO — Simulateur de variance
═══════════════════════════════════════════════════════

Même avec un edge réel, une série de paris connaît des
trajectoires très dispersées : des pertes prolongées sont
NORMALES. Ce module simule des milliers de trajectoires à
partir des paris réellement enregistrés pour montrer :

- la distribution des profits possibles,
- le drawdown maximal auquel s'attendre,
- la probabilité d'être perdant après N paris MÊME si le
  modèle a raison,

sous deux hypothèses : « le modèle a raison » (les paris
gagnent avec la probabilité du modèle) et « le marché a
raison » (aucun edge, on paie la marge).
"""

from typing import Dict, List

import numpy as np

# Marge typique payée quand le marché a raison (bookmaker soft)
DEFAULT_MARGIN = 0.06


def simulate_bets(bets: List[Dict], n_sims: int = 10_000,
                  seed: int = 42) -> Dict:
    """
    Simule n_sims trajectoires de la séquence de paris fournie.

    Args:
        bets: liste de dicts avec au minimum
              bookmaker_odds, recommended_stake, model_probability
    Returns:
        {"n_paris", "total_mise", "modele_correct": {...},
         "marche_correct": {...}} — ou {} si pas assez de paris.
    """

    usable = [
        b for b in bets
        if (b.get("bookmaker_odds") or 0) > 1
        and (b.get("recommended_stake") or 0) > 0
        and 0 < (b.get("model_probability") or 0) < 1
    ]

    if len(usable) < 5:
        return {}

    odds = np.array([b["bookmaker_odds"] for b in usable])
    stakes = np.array([b["recommended_stake"] for b in usable])
    p_model = np.array([b["model_probability"] for b in usable])

    # Hypothèse « marché correct » : l'EV de chaque pari = −marge
    p_market = np.clip((1 - DEFAULT_MARGIN) / odds, 0.01, 0.99)

    rng = np.random.default_rng(seed)

    def run(p: np.ndarray) -> Dict:
        # matrice (n_sims × n_paris) de gains/pertes
        wins = rng.random((n_sims, len(usable))) < p
        profits = np.where(wins, stakes * (odds - 1), -stakes)

        cumul = np.cumsum(profits, axis=1)
        finals = cumul[:, -1]

        # Drawdown maximal de chaque trajectoire
        running_max = np.maximum.accumulate(
            np.concatenate([np.zeros((n_sims, 1)), cumul], axis=1), axis=1
        )
        drawdowns = (running_max[:, 1:] - cumul).max(axis=1)

        return {
            "profit_moyen": float(finals.mean()),
            "percentiles": {
                "p5": float(np.percentile(finals, 5)),
                "p25": float(np.percentile(finals, 25)),
                "p50": float(np.percentile(finals, 50)),
                "p75": float(np.percentile(finals, 75)),
                "p95": float(np.percentile(finals, 95)),
            },
            "prob_perte": float((finals < 0).mean()),
            "drawdown_median": float(np.percentile(drawdowns, 50)),
            "drawdown_p95": float(np.percentile(drawdowns, 95)),
            "distribution": np.histogram(finals, bins=40),
        }

    return {
        "n_paris": len(usable),
        "total_mise": float(stakes.sum()),
        "modele_correct": run(p_model),
        "marche_correct": run(p_market),
    }
