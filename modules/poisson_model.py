"""
═══════════════════════════════════════════════════════
 MODULE POISSON — Modèle de prédiction par la loi
 de Poisson bivariée (scores de football)
═══════════════════════════════════════════════════════

Le nombre de buts d'une équipe suit approximativement une
loi de Poisson. On estime λ_domicile et λ_extérieur, puis
on construit la matrice des scores pour en déduire les
probabilités 1X2, Over/Under et BTTS.

Les λ sont ancrés sur le marché (cotes no-vig) puis ajustés
avec les stats/xG disponibles — le marché reste la meilleure
estimation de base, les stats servent à détecter les écarts.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import PoissonConfig
from modules.data_collector import MatchContext
from modules.odds_utils import novig_probs, margin_ok


# ══════════════════════════════════════════════════════
#  RÉSULTAT DE PRÉDICTION
# ══════════════════════════════════════════════════════

@dataclass
class PoissonPrediction:
    """Résultat complet d'une prédiction Poisson."""

    lambda_home: float = 1.30
    lambda_away: float = 1.10

    # 1X2
    prob_home: float = 0.0
    prob_draw: float = 0.0
    prob_away: float = 0.0

    # Buts
    prob_over15: float = 0.0
    prob_under15: float = 0.0
    prob_over25: float = 0.0
    prob_under25: float = 0.0
    prob_over35: float = 0.0
    prob_under35: float = 0.0

    # BTTS
    prob_btts_yes: float = 0.0
    prob_btts_no: float = 0.0

    # Totaux par équipe — {"0_5": P(over), "1_5": ..., "2_5": ...}
    team_totals_home: Dict[str, float] = field(default_factory=dict)
    team_totals_away: Dict[str, float] = field(default_factory=dict)

    # Buts par mi-temps (match entier) — {"0_5": P(over), "1_5": ...}
    h1_totals: Dict[str, float] = field(default_factory=dict)
    h2_totals: Dict[str, float] = field(default_factory=dict)

    # Buts par mi-temps ET par équipe — {"0_5": P(over), "1_5": ...}
    h1_team_home: Dict[str, float] = field(default_factory=dict)
    h1_team_away: Dict[str, float] = field(default_factory=dict)
    h2_team_home: Dict[str, float] = field(default_factory=dict)
    h2_team_away: Dict[str, float] = field(default_factory=dict)

    # Tirs cadrés attendus par équipe (λ, approximation depuis les buts)
    sot_lambda_home: float = 0.0
    sot_lambda_away: float = 0.0
    sot_lambda_total: float = 0.0

    # Double chance
    prob_1x: float = 0.0
    prob_x2: float = 0.0
    prob_12: float = 0.0

    # Scores
    most_likely_score: str = "1-1"
    top_scores: List[Tuple[str, float]] = field(default_factory=list)
    score_matrix: List[List[float]] = field(default_factory=list)

    confidence: float = 50.0
    model_name: str = "Poisson"


# ══════════════════════════════════════════════════════
#  PRÉDICTEUR
# ══════════════════════════════════════════════════════

class PoissonPredictor:
    """Modèle de Poisson calibré marché + stats."""

    def predict(self, context: MatchContext,
                lambda_multipliers: Tuple[float, float] = (1.0, 1.0)
                ) -> PoissonPrediction:
        """
        Prédit toutes les probabilités d'un match.

        lambda_multipliers : ajustements (domicile, extérieur) issus de
        l'analyse pré-match (H2H, forme, contexte) — bornés par sécurité.
        """

        lam_home, lam_away = self._estimate_lambdas(context)

        # Ajustements du conseil d'agents (bornés une seconde fois)
        mult_h = max(0.85, min(1.15, lambda_multipliers[0]))
        mult_a = max(0.85, min(1.15, lambda_multipliers[1]))
        lam_home = max(PoissonConfig.MIN_LAMBDA,
                       min(PoissonConfig.MAX_LAMBDA, lam_home * mult_h))
        lam_away = max(PoissonConfig.MIN_LAMBDA,
                       min(PoissonConfig.MAX_LAMBDA, lam_away * mult_a))

        pred = PoissonPrediction(
            lambda_home=round(lam_home, 3),
            lambda_away=round(lam_away, 3),
        )

        self._fill_probabilities(
            pred,
            first_half_share=getattr(context, "first_half_share", None),
            context=context,
        )

        # Confiance selon la qualité des données
        pred.confidence = min(40 + context.data_completeness * 0.5, 90)

        return pred

    # ─── ESTIMATION DES LAMBDAS ─────────────────────

    def _estimate_lambdas(self, context: MatchContext) -> Tuple[float, float]:
        """
        Estime λ_home et λ_away en combinant :
        1. Le marché (cotes no-vig 1X2 + over/under)  — poids MARKET_WEIGHT
        2. Les stats des équipes (buts moyens ou xG)  — poids restant
        """

        stats_lams = self._lambdas_from_stats(context)
        market_lams = self._lambdas_from_market(context)

        if market_lams:
            w = PoissonConfig.MARKET_WEIGHT
            lam_home = w * market_lams[0] + (1 - w) * stats_lams[0]
            lam_away = w * market_lams[1] + (1 - w) * stats_lams[1]
        else:
            lam_home, lam_away = stats_lams

        # Bornes de sécurité
        lam_home = max(PoissonConfig.MIN_LAMBDA,
                       min(PoissonConfig.MAX_LAMBDA, lam_home))
        lam_away = max(PoissonConfig.MIN_LAMBDA,
                       min(PoissonConfig.MAX_LAMBDA, lam_away))

        return lam_home, lam_away

    def _lambdas_from_stats(self, context: MatchContext) -> Tuple[float, float]:
        """
        λ depuis les stats des équipes (xG en priorité).

        Trois natures de stats, combinées côté par côté :
        - "réelles" (API, historique, xG) : moyennes vs adversaires
          moyens → formule multiplicative attaque × défense / moyenne
        - "estimées" (dérivées des cotes) : déjà spécifiques à CE
          matchup → les multiplier compterait l'écart de niveau deux
          fois, on moyenne les deux estimations de la même quantité
        L'avantage domicile n'est réappliqué que sur les stats sans
        déclinaison domicile/extérieur (xG), jamais sur les splits
        domicile/extérieur qui l'incluent déjà.
        """

        home = context.home_stats
        away = context.away_stats
        league_avg = context.league_avg_goals / 2  # buts moyens par équipe

        def side_profile(stats, is_home):
            """(attaque, défense, estimé?, avantage domicile déjà inclus?)"""
            if stats and stats.xg_available and stats.xg_for_home > 0:
                # xG (splits domicile/extérieur) MÉLANGÉ aux buts réels.
                # Le xG est plus prédictif (moins de bruit), mais les
                # buts capturent la finition réelle : le mélange est
                # plus robuste que le xG seul (best practice). Les
                # splits incluent déjà l'avantage du terrain.
                b = PoissonConfig.XG_BLEND
                if is_home:
                    xg_a, xg_d = stats.xg_for_home, stats.xg_against_home
                    g_a, g_d = (stats.avg_goals_scored_home,
                                stats.avg_goals_conceded_home)
                else:
                    xg_a, xg_d = stats.xg_for_away, stats.xg_against_away
                    g_a, g_d = (stats.avg_goals_scored_away,
                                stats.avg_goals_conceded_away)
                # Si les splits de buts manquent, xG seul
                if not (g_a and g_d):
                    g_a, g_d = xg_a, xg_d
                att = b * xg_a + (1 - b) * g_a
                deff = b * xg_d + (1 - b) * g_d
                return att, deff, stats.data_source == "estimated", True
            if stats and stats.xg_available:
                # xG sans split : moyenne toutes venues → avantage
                # domicile à appliquer ensuite
                return stats.xg_scored, stats.xg_conceded, \
                    stats.data_source == "estimated", False
            if stats:
                if is_home:
                    return stats.avg_goals_scored_home, \
                        stats.avg_goals_conceded_home, \
                        stats.data_source == "estimated", True
                return stats.avg_goals_scored_away, \
                    stats.avg_goals_conceded_away, \
                    stats.data_source == "estimated", True
            # Aucune donnée : profil moyen de la ligue
            if is_home:
                return league_avg * 1.1, league_avg * 0.95, True, True
            return league_avg * 0.9, league_avg * 1.05, True, True

        h_att, h_def, h_est, h_ha = side_profile(home, is_home=True)
        a_att, a_def, a_est, a_ha = side_profile(away, is_home=False)

        def combine(attack, opp_defense, att_est, def_est):
            """Deux estimations de la même quantité (buts de ce côté)."""
            if att_est or def_est:
                # Au moins une est déjà spécifique au matchup → moyenne
                return (attack + opp_defense) / 2
            # Deux moyennes vs adversaires moyens → multiplicative
            return attack * opp_defense / max(league_avg, 0.5)

        lam_home = combine(h_att, a_def, h_est, a_est)
        lam_away = combine(a_att, h_def, a_est, h_est)

        # Avantage domicile : uniquement si pas déjà dans les splits
        if not h_ha:
            lam_home *= PoissonConfig.HOME_ADVANTAGE
        if not a_ha:
            lam_away *= (2 - PoissonConfig.HOME_ADVANTAGE)

        return lam_home, lam_away

    def _lambdas_from_market(self,
                              context: MatchContext
                              ) -> Optional[Tuple[float, float]]:
        """
        λ implicites du marché : on cherche (λh, λa) tels que la
        matrice de Poisson reproduise au mieux les probabilités
        no-vig 1X2 (et le total over/under 2.5 si disponible).
        """

        odds = context.odds or {}
        o1 = float(odds.get("1", 0) or 0)
        ox = float(odds.get("X", 0) or 0)
        o2 = float(odds.get("2", 0) or 0)

        if o1 <= 1 or ox <= 1 or o2 <= 1:
            return None

        # Cotes corrompues (OCR) : ne JAMAIS s'ancrer dessus —
        # sinon tous les marchés héritent d'une ancre fausse
        if not margin_ok([o1, ox, o2]):
            return None

        probs = novig_probs([o1, ox, o2])
        if not probs:
            return None
        target_1, target_x, target_2 = probs

        # Total de buts cible
        o_over = float(odds.get("over_2_5", 0) or 0)
        o_under = float(odds.get("under_2_5", 0) or 0)
        if o_over > 1 and o_under > 1 and margin_ok([o_over, o_under]):
            p_over = novig_probs([o_over, o_under])[0]
            total_candidates = [self._invert_over25(p_over)]
        else:
            # balayer plusieurs totaux autour de la moyenne de la ligue
            base = context.league_avg_goals
            total_candidates = [base - 0.4, base - 0.2, base,
                                base + 0.2, base + 0.4]

        # Recherche par grille : répartition du total entre les équipes
        best = None
        best_err = float("inf")

        for total_goals in total_candidates:
            total_goals = max(1.0, min(5.5, total_goals))
            for share in [x / 100 for x in range(25, 76)]:
                lh = total_goals * share
                la = total_goals * (1 - share)
                p1, px, p2 = self._quick_1x2(lh, la)
                err = ((p1 - target_1) ** 2 +
                       (px - target_x) ** 2 +
                       (p2 - target_2) ** 2)
                if err < best_err:
                    best_err = err
                    best = (lh, la)

        return best

    # ─── PROBABILITÉS ───────────────────────────────

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        return math.exp(-lam) * lam ** k / math.factorial(k)

    @classmethod
    def _score_matrix(cls, lam_home: float,
                      lam_away: float) -> List[List[float]]:
        """
        Matrice des probabilités de score avec correction Dixon-Coles.

        Le Poisson indépendant sous-estime les 0-0 et 1-1 et surestime
        les 1-0/0-1. Dixon & Coles (1997) corrigent les 4 cases basses
        par un facteur tau (rho négatif => 0-0 et 1-1 gonflés) :
          tau(0,0) = 1 − λμρ    tau(0,1) = 1 + λρ
          tau(1,0) = 1 + μρ     tau(1,1) = 1 − ρ
        La matrice est ensuite renormalisée.
        """

        max_g = PoissonConfig.MAX_GOALS
        rho = PoissonConfig.DIXON_COLES_RHO

        ph = [cls._poisson_pmf(i, lam_home) for i in range(max_g + 1)]
        pa = [cls._poisson_pmf(j, lam_away) for j in range(max_g + 1)]

        matrix = [[ph[i] * pa[j] for j in range(max_g + 1)]
                  for i in range(max_g + 1)]

        matrix[0][0] *= max(0.0, 1 - lam_home * lam_away * rho)
        matrix[0][1] *= max(0.0, 1 + lam_home * rho)
        matrix[1][0] *= max(0.0, 1 + lam_away * rho)
        matrix[1][1] *= max(0.0, 1 - rho)

        total = sum(sum(row) for row in matrix)
        if total > 0:
            matrix = [[p / total for p in row] for row in matrix]

        return matrix

    def _quick_1x2(self, lam_home: float,
                   lam_away: float) -> Tuple[float, float, float]:
        """1X2 rapide depuis deux lambdas (matrice Dixon-Coles)."""

        matrix = self._score_matrix(lam_home, lam_away)
        max_g = PoissonConfig.MAX_GOALS

        p1 = px = p2 = 0.0
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                p = matrix[i][j]
                if i > j:
                    p1 += p
                elif i == j:
                    px += p
                else:
                    p2 += p

        return p1, px, p2

    def _fill_probabilities(self, pred: PoissonPrediction,
                            first_half_share: float = None,
                            context=None):
        """Remplit toutes les probabilités depuis la matrice des scores."""

        max_g = PoissonConfig.MAX_GOALS
        lam_h, lam_a = pred.lambda_home, pred.lambda_away

        matrix = self._score_matrix(lam_h, lam_a)
        pred.score_matrix = matrix

        p1 = px = p2 = 0.0
        over15 = over25 = over35 = 0.0
        btts = 0.0
        scores = []

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                p = matrix[i][j]
                scores.append((f"{i}-{j}", p))

                if i > j:
                    p1 += p
                elif i == j:
                    px += p
                else:
                    p2 += p

                total = i + j
                if total >= 2:
                    over15 += p
                if total >= 3:
                    over25 += p
                if total >= 4:
                    over35 += p

                if i >= 1 and j >= 1:
                    btts += p

        # La matrice Dixon-Coles est déjà normalisée
        norm = p1 + px + p2
        if norm > 0:
            p1, px, p2 = p1 / norm, px / norm, p2 / norm

        pred.prob_home = round(p1, 4)
        pred.prob_draw = round(px, 4)
        pred.prob_away = round(p2, 4)

        pred.prob_over15 = round(over15, 4)
        pred.prob_under15 = round(1 - over15, 4)
        pred.prob_over25 = round(over25, 4)
        pred.prob_under25 = round(1 - over25, 4)
        pred.prob_over35 = round(over35, 4)
        pred.prob_under35 = round(1 - over35, 4)

        pred.prob_btts_yes = round(btts, 4)
        pred.prob_btts_no = round(1 - btts, 4)

        pred.prob_1x = round(p1 + px, 4)
        pred.prob_x2 = round(px + p2, 4)
        pred.prob_12 = round(p1 + p2, 4)

        scores.sort(key=lambda s: s[1], reverse=True)
        pred.top_scores = [(s, round(p, 4)) for s, p in scores[:5]]
        pred.most_likely_score = scores[0][0] if scores else "1-1"

        # ── Totaux par équipe (loi de Poisson exacte sur chaque λ) ──
        pred.team_totals_home = self._over_probs(lam_h, ("0_5", "1_5", "2_5"))
        pred.team_totals_away = self._over_probs(lam_a, ("0_5", "1_5", "2_5"))

        # ── Buts par mi-temps (part 1MT propre à la ligue) ──
        share = first_half_share or PoissonConfig.FIRST_HALF_SHARE
        lam_total = lam_h + lam_a
        lam_h1 = lam_total * share
        lam_h2 = lam_total * (1 - share)
        # Lignes 0,5 / 1,5 / 2,5 : celles réellement proposées par
        # Betclic sur les marchés de mi-temps
        half_lines = ("0_5", "1_5", "2_5")
        pred.h1_totals = self._over_probs(lam_h1, half_lines)
        pred.h2_totals = self._over_probs(lam_h2, half_lines)

        # ── Buts par mi-temps ET par équipe (λ équipe × part MT) ──
        pred.h1_team_home = self._over_probs(lam_h * share, half_lines)
        pred.h1_team_away = self._over_probs(lam_a * share, half_lines)
        pred.h2_team_home = self._over_probs(
            lam_h * (1 - share), half_lines)
        pred.h2_team_away = self._over_probs(
            lam_a * (1 - share), half_lines)

        # ── Tirs cadrés attendus ──
        # Les tirs cadrés ne se déduisent PAS des buts : le ratio
        # tirs/but varie de 2.95 (Bundesliga) à 3.32 (Serie A) —
        # les championnats qui marquent peu tirent quand même. Quand
        # les vraies moyennes sont disponibles, on croise attaque et
        # défense comme pour les buts ; sinon seulement, repli sur
        # l'approximation.
        sot = self._sot_lambdas(context)
        if sot:
            pred.sot_lambda_home, pred.sot_lambda_away = sot
        else:
            pred.sot_lambda_home = round(
                lam_h * PoissonConfig.SOT_PER_GOAL, 2)
            pred.sot_lambda_away = round(
                lam_a * PoissonConfig.SOT_PER_GOAL, 2)
        pred.sot_lambda_total = round(
            pred.sot_lambda_home + pred.sot_lambda_away, 2)

    @staticmethod
    def _sot_lambdas(context) -> Optional[Tuple[float, float]]:
        """
        Tirs cadrés attendus de chaque équipe, depuis les moyennes
        RÉELLES : attaque de l'une × défense de l'autre, normalisées
        par la moyenne de la ligue.

            λ = tirs_cadrés_pour(A) × tirs_cadrés_encaissés(B)
                / moyenne_ligue

        None si les données réelles manquent (ligue non couverte,
        équipe inconnue, réseau indisponible).
        """

        if context is None:
            return None

        home = getattr(context, "home_stats", None)
        away = getattr(context, "away_stats", None)
        ligue = float(getattr(context, "league_avg_sot", 0) or 0)

        if (home is None or away is None or ligue <= 0
                or not getattr(home, "sot_available", False)
                or not getattr(away, "sot_available", False)):
            return None

        lam_h = home.avg_sot_for * away.avg_sot_against / ligue
        lam_a = away.avg_sot_for * home.avg_sot_against / ligue

        # Bornes de sécurité : une équipe reste dans le plausible
        lam_h = min(max(lam_h, 1.0), 12.0)
        lam_a = min(max(lam_a, 1.0), 12.0)
        return round(lam_h, 2), round(lam_a, 2)

    @classmethod
    def fit_lambda_from_over(cls, p_over: float, line: str,
                             lo: float = 0.05, hi: float = 40.0
                             ) -> Optional[float]:
        """
        λ tel que P(N > ligne) = p_over pour N ~ Poisson(λ).

        Inversion par bissection (P(N > ligne) est strictement
        croissante en λ). Sert à lire le λ que le marché implique
        sur un marché où le modèle n'a aucune donnée propre.
        Retourne None si p_over n'est pas une probabilité utile.
        """

        if not (0.001 < p_over < 0.999):
            return None

        for _ in range(60):
            mid = (lo + hi) / 2
            if cls.poisson_over(mid, line) < p_over:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 3)

    @classmethod
    def _over_probs(cls, lam: float, lines: tuple) -> Dict[str, float]:
        """P(N > ligne) pour N ~ Poisson(λ), lignes au format "1_5"."""

        return {
            line: round(cls.poisson_over(lam, line), 4)
            for line in lines
        }

    @staticmethod
    def poisson_over(lam: float, line: str) -> float:
        """
        P(N ≥ k) pour N ~ Poisson(λ), où k = ligne arrondie au-dessus
        (ligne "2_5" → P(N ≥ 3)).
        """

        threshold = int(float(line.replace("_", "."))) + 1
        cumulative = 0.0
        term = math.exp(-lam)
        for k in range(threshold):
            if k > 0:
                term *= lam / k
            cumulative += term
        return 1 - cumulative

    @staticmethod
    def _invert_over25(p_over: float) -> float:
        """Trouve λ_total tel que P(N ≥ 3) = p_over (N ~ Poisson)."""

        p_over = max(0.02, min(0.98, p_over))

        def prob_over(lam: float) -> float:
            p0 = math.exp(-lam)
            p1 = p0 * lam
            p2 = p1 * lam / 2
            return 1 - (p0 + p1 + p2)

        lo, hi = 0.2, 6.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if prob_over(mid) < p_over:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2
