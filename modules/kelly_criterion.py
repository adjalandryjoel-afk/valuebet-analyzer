"""
═══════════════════════════════════════════════════════
 MODULE KELLY — Calcul des mises optimales
 (critère de Kelly fractionné)
═══════════════════════════════════════════════════════

Kelly plein : f* = (b·p - q) / b
  où b = cote - 1, p = prob de gain, q = 1 - p

On applique un quart de Kelly (KELLY_FRACTION) avec un
plafond par pari et un plafond d'exposition totale — la
variance du Kelly plein est trop violente pour un bankroll
réel.
"""

from typing import List
from dataclasses import dataclass, field

from config import KellyConfig
from modules.value_detector import ValueBet, MatchAnalysis


@dataclass
class StakeRecommendation:
    """Recommandation de mise pour un value bet."""

    value_bet: ValueBet = None

    kelly_fraction: float = 0.0      # Kelly plein (fraction du bankroll)
    fractional_kelly: float = 0.0    # Kelly fractionné appliqué
    stake_percentage: float = 0.0    # % du bankroll misé
    stake_amount: float = 0.0        # montant en FCFA

    potential_profit: float = 0.0    # gain net si le pari passe
    risk_level: str = ""             # 🟢 Faible / 🟡 Moyen / 🔴 Élevé


class KellyStakeCalculator:
    """Calcule les mises Kelly fractionnées pour les value bets."""

    def __init__(self, bankroll: float = None):
        self.bankroll = bankroll or KellyConfig.DEFAULT_BANKROLL

    # ─── CALCUL PAR MATCH ───────────────────────────

    def calculate_stake(self, vb: ValueBet) -> StakeRecommendation:
        """Calcule la mise recommandée pour un value bet."""

        rec = StakeRecommendation(value_bet=vb)

        b = vb.bookmaker_odds - 1
        p = vb.model_probability
        q = 1 - p

        if b <= 0:
            return rec

        kelly_full = (b * p - q) / b
        kelly_full = max(0.0, kelly_full)

        fractional = kelly_full * KellyConfig.KELLY_FRACTION

        # Plafond par pari
        stake_pct = min(fractional * 100, KellyConfig.MAX_STAKE_PERCENTAGE)

        # Montant arrondi
        amount = self.bankroll * stake_pct / 100
        rounding = KellyConfig.STAKE_ROUNDING
        amount = round(amount / rounding) * rounding

        # Mise minimum
        if amount < KellyConfig.MIN_STAKE_AMOUNT:
            amount = 0.0
            stake_pct = 0.0

        rec.kelly_fraction = round(kelly_full, 4)
        rec.fractional_kelly = round(fractional, 4)
        rec.stake_percentage = round(stake_pct, 2)
        rec.stake_amount = amount
        rec.potential_profit = round(amount * b, 0)

        # Niveau de risque selon la cote et la probabilité
        if vb.bookmaker_odds <= 2.0 and p >= 0.50:
            rec.risk_level = "🟢 Faible"
        elif vb.bookmaker_odds <= 3.5:
            rec.risk_level = "🟡 Moyen"
        else:
            rec.risk_level = "🔴 Élevé"

        # Répercuter sur le ValueBet (utilisé par l'UI et la DB)
        vb.kelly_stake = rec.stake_percentage
        vb.recommended_stake = amount

        return rec

    def calculate_all_stakes(self,
                             analysis: MatchAnalysis
                             ) -> List[StakeRecommendation]:
        """Calcule les mises pour tous les value bets d'un match."""

        return [self.calculate_stake(vb) for vb in analysis.value_bets]

    # ─── AJUSTEMENT PORTFOLIO ───────────────────────

    def portfolio_adjustment(self,
                             stakes: List[StakeRecommendation]
                             ) -> List[StakeRecommendation]:
        """
        Réduit proportionnellement toutes les mises si l'exposition
        totale dépasse MAX_TOTAL_EXPOSURE % du bankroll.
        """

        total = sum(s.stake_amount for s in stakes)
        max_exposure = self.bankroll * KellyConfig.MAX_TOTAL_EXPOSURE / 100

        if total <= max_exposure or total <= 0:
            return stakes

        scale = max_exposure / total
        rounding = KellyConfig.STAKE_ROUNDING

        for s in stakes:
            new_amount = s.stake_amount * scale
            new_amount = round(new_amount / rounding) * rounding

            if new_amount < KellyConfig.MIN_STAKE_AMOUNT:
                new_amount = 0.0

            s.stake_amount = new_amount
            s.stake_percentage = round(
                new_amount / self.bankroll * 100, 2
            )
            s.potential_profit = round(
                new_amount * (s.value_bet.bookmaker_odds - 1), 0
            )

            # Répercuter sur le ValueBet
            s.value_bet.recommended_stake = new_amount
            s.value_bet.kelly_stake = s.stake_percentage

        return stakes
