"""
═══════════════════════════════════════════════════════
 MODULE VALUE DETECTOR — Détection des value bets
═══════════════════════════════════════════════════════

Un value bet existe quand :
    probabilité_modèle × cote_bookmaker > 1

On compare les probabilités du modèle (blend Poisson + Elo)
aux probabilités implicites des cotes Betclic, marché par
marché, et on ne retient que les écarts au-dessus des seuils
de config (value minimum, confiance minimum, cotes jouables).
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from config import ValueBetConfig, KellyConfig


# ══════════════════════════════════════════════════════
#  STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class ValueBet:
    """Un value bet détecté."""

    match: str = ""
    market: str = ""
    selection: str = ""

    bookmaker_odds: float = 0.0
    fair_odds: float = 0.0

    model_probability: float = 0.0
    implied_probability: float = 0.0

    value_percentage: float = 0.0   # fraction : 0.08 = +8%
    edge: float = 0.0               # prob modèle - prob implicite

    confidence_score: float = 0.0   # 0-100
    value_rating: str = ""          # ⭐ à ⭐⭐⭐⭐⭐

    # Remplis par le KellyStakeCalculator
    kelly_stake: float = 0.0        # % du bankroll
    recommended_stake: float = 0.0  # montant FCFA


@dataclass
class MatchAnalysis:
    """Analyse complète d'un match."""

    home_team: str = ""
    away_team: str = ""
    competition: str = ""

    # Cotes Betclic (clés "1", "X", "2", "over_2_5", ...)
    odds: Dict = field(default_factory=dict)

    # Probabilités du modèle par marché
    # {"1X2": {"1":…, "X":…, "2":…}, "OU25": {...}, "BTTS": {...}}
    model_probs: Dict = field(default_factory=dict)

    # Value bets détectés
    value_bets: List[ValueBet] = field(default_factory=list)
    has_value: bool = False
    total_value_bets: int = 0

    # Prédiction
    predicted_score: str = ""
    most_likely_result: str = ""

    # Marché
    bookmaker_margin: float = 0.0   # en %

    # Qualité
    analysis_confidence: float = 0.0  # 0-100
    data_warning: str = ""


# ══════════════════════════════════════════════════════
#  DÉTECTEUR
# ══════════════════════════════════════════════════════

class ValueBetDetector:
    """Compare le modèle au marché et signale les values."""

    def analyze_match(self, home_team: str, away_team: str,
                      odds: Dict, poisson_pred, elo_pred,
                      competition: str = "",
                      min_value: float = None,
                      min_confidence: float = None) -> MatchAnalysis:
        """
        Analyse tous les marchés d'un match.

        Args:
            odds: cotes Betclic {"1", "X", "2", "over_2_5",
                  "under_2_5", "btts_oui", "btts_non"}
            poisson_pred: PoissonPrediction
            elo_pred: EloPrediction
            min_value / min_confidence : seuils de l'interface
                (défaut : valeurs de config)
        """

        # Seuils effectifs — passés en paramètres aux évaluations
        # (jamais stockés sur l'instance : elle est partagée entre
        # les sessions Streamlit via st.cache_resource)
        min_value_eff = (min_value if min_value is not None
                         else ValueBetConfig.MIN_VALUE_THRESHOLD)
        min_conf_eff = (min_confidence if min_confidence is not None
                        else ValueBetConfig.MIN_CONFIDENCE_SCORE)

        analysis = MatchAnalysis(
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            odds=dict(odds or {}),
        )

        match_label = f"{home_team} vs {away_team}"

        # ── Blend des probabilités 1X2 (Poisson + Elo) ──
        wp = ValueBetConfig.POISSON_WEIGHT
        we = ValueBetConfig.ELO_WEIGHT

        p1 = wp * poisson_pred.prob_home + we * elo_pred.prob_home_win
        px = wp * poisson_pred.prob_draw + we * elo_pred.prob_draw
        p2 = wp * poisson_pred.prob_away + we * elo_pred.prob_away_win

        # Normaliser
        total = p1 + px + p2
        if total > 0:
            p1, px, p2 = p1 / total, px / total, p2 / total

        analysis.model_probs["1X2"] = {
            "1": round(p1, 4), "X": round(px, 4), "2": round(p2, 4)
        }
        analysis.model_probs["OU25"] = {
            "over": poisson_pred.prob_over25,
            "under": poisson_pred.prob_under25,
        }
        analysis.model_probs["BTTS"] = {
            "yes": poisson_pred.prob_btts_yes,
            "no": poisson_pred.prob_btts_no,
        }

        # ── Prédiction texte ──
        analysis.predicted_score = poisson_pred.most_likely_score
        best = max([("1", p1), ("X", px), ("2", p2)], key=lambda t: t[1])
        analysis.most_likely_result = {
            "1": f"Victoire {home_team}",
            "X": "Match nul",
            "2": f"Victoire {away_team}",
        }[best[0]]

        # ── Marge du bookmaker (sur le 1X2) ──
        o1 = float(odds.get("1", 0) or 0)
        ox = float(odds.get("X", 0) or 0)
        o2 = float(odds.get("2", 0) or 0)

        odds_1x2_suspect = False
        if o1 > 1 and ox > 1 and o2 > 1:
            analysis.bookmaker_margin = round(
                (1 / o1 + 1 / ox + 1 / o2 - 1) * 100, 2
            )
            # Marge anormale (négative ou énorme) = cotes 1X2 incohérentes,
            # très probablement une erreur de lecture de la capture
            if not (0 <= analysis.bookmaker_margin
                    <= ValueBetConfig.MAX_SANE_MARGIN):
                odds_1x2_suspect = True

        # ── Accord entre les modèles (pour la confiance) ──
        model_agreement = 1 - (
            abs(poisson_pred.prob_home - elo_pred.prob_home_win)
            + abs(poisson_pred.prob_away - elo_pred.prob_away_win)
        ) / 2

        # ── Scanner chaque marché ──
        # (marché, sélection, cote, prob modèle, pénalité de confiance)
        candidates = [
            ("1X2", "1", o1, p1, 0),
            ("1X2", "X", ox, px, 0),
            ("1X2", "2", o2, p2, 0),
            ("Over/Under 2.5", "Over 2.5",
             float(odds.get("over_2_5", 0) or 0), poisson_pred.prob_over25, 0),
            ("Over/Under 2.5", "Under 2.5",
             float(odds.get("under_2_5", 0) or 0), poisson_pred.prob_under25, 0),
            ("BTTS", "Oui",
             float(odds.get("btts_oui", 0) or 0), poisson_pred.prob_btts_yes, 0),
            ("BTTS", "Non",
             float(odds.get("btts_non", 0) or 0), poisson_pred.prob_btts_no, 0),
        ]

        # ── Totaux de buts par équipe ──
        team_sides = (
            ("home", home_team, poisson_pred.team_totals_home, "HOME_TOTALS"),
            ("away", away_team, poisson_pred.team_totals_away, "AWAY_TOTALS"),
        )
        for side, team, totals, probs_key in team_sides:
            analysis.model_probs[probs_key] = {}
            for line, p_over in totals.items():
                line_txt = line.replace("_", ".")
                analysis.model_probs[probs_key][f"over_{line}"] = round(p_over, 4)
                analysis.model_probs[probs_key][f"under_{line}"] = round(1 - p_over, 4)
                candidates.append((
                    f"Buts {team}", f"Over {line_txt}",
                    float(odds.get(f"{side}_over_{line}", 0) or 0), p_over, 0))
                candidates.append((
                    f"Buts {team}", f"Under {line_txt}",
                    float(odds.get(f"{side}_under_{line}", 0) or 0), 1 - p_over, 0))

        # ── Buts par mi-temps (répartition 45/55 → légère pénalité) ──
        halves = (
            ("h1", "1ère mi-temps", poisson_pred.h1_totals, "H1"),
            ("h2", "2ème mi-temps", poisson_pred.h2_totals, "H2"),
        )
        for prefix, label, totals, probs_key in halves:
            analysis.model_probs[probs_key] = {}
            for line, p_over in totals.items():
                line_txt = line.replace("_", ".")
                analysis.model_probs[probs_key][f"over_{line}"] = round(p_over, 4)
                analysis.model_probs[probs_key][f"under_{line}"] = round(1 - p_over, 4)
                candidates.append((
                    f"Buts {label}", f"Over {line_txt}",
                    float(odds.get(f"{prefix}_over_{line}", 0) or 0), p_over, 5))
                candidates.append((
                    f"Buts {label}", f"Under {line_txt}",
                    float(odds.get(f"{prefix}_under_{line}", 0) or 0), 1 - p_over, 5))

        # ── Buts par équipe ET par mi-temps (λ équipe × part MT
        #    → pénalité intermédiaire) ──
        team_halves = (
            ("h1_home", f"Buts 1MT {home_team}",
             poisson_pred.h1_team_home, "H1_HOME"),
            ("h1_away", f"Buts 1MT {away_team}",
             poisson_pred.h1_team_away, "H1_AWAY"),
            ("h2_home", f"Buts 2MT {home_team}",
             poisson_pred.h2_team_home, "H2_HOME"),
            ("h2_away", f"Buts 2MT {away_team}",
             poisson_pred.h2_team_away, "H2_AWAY"),
        )
        for prefix, market_label, totals, probs_key in team_halves:
            analysis.model_probs[probs_key] = {}
            for line, p_over in totals.items():
                line_txt = line.replace("_", ".")
                analysis.model_probs[probs_key][
                    f"over_{line}"] = round(p_over, 4)
                analysis.model_probs[probs_key][
                    f"under_{line}"] = round(1 - p_over, 4)
                candidates.append((
                    market_label, f"Over {line_txt}",
                    float(odds.get(f"{prefix}_over_{line}", 0) or 0),
                    p_over, 8))
                candidates.append((
                    market_label, f"Under {line_txt}",
                    float(odds.get(f"{prefix}_under_{line}", 0) or 0),
                    1 - p_over, 8))

        # ── Tirs cadrés (équipe ou match) ──
        # Le modèle n'a AUCUNE donnée propre sur les tirs : il ne sait
        # que les déduire des buts attendus (× SOT_PER_GOAL), une
        # approximation qui dérive vite du réel. Le bookmaker, lui,
        # dispose des vraies stats de tirs. On ancre donc le λ au
        # marché (même principe que MARKET_WEIGHT sur les buts,
        # calibré par backtest) : sans cet ancrage, l'écart entre
        # l'approximation et le marché se lit à tort comme de la
        # value et produit des paris fantômes sur chaque match.
        from modules.poisson_model import PoissonPredictor
        from modules.odds_utils import novig_probs
        from config import PoissonConfig

        analysis.model_probs["SOT_HOME"] = {}
        analysis.model_probs["SOT_AWAY"] = {}
        analysis.model_probs["SOT_TOTAL"] = {}
        sot_pattern = re.compile(
            r"^sot_(home|away|total)_(over|under)_(\d+_5)$")

        # 1) Regrouper les cotes de tirs cadrés par côté et par ligne
        sot_odds: Dict[str, Dict[str, Dict[str, float]]] = {}
        for key, value in odds.items():
            m = sot_pattern.match(str(key))
            if not m:
                continue
            side, direction, line = m.groups()
            try:
                cote = float(value or 0)
            except (TypeError, ValueError):
                continue
            if cote > 1:
                sot_odds.setdefault(side, {}).setdefault(
                    line, {})[direction] = cote

        proxy = {
            "home": poisson_pred.sot_lambda_home,
            "away": poisson_pred.sot_lambda_away,
            "total": getattr(poisson_pred, "sot_lambda_total", 0)
            or (poisson_pred.sot_lambda_home
                + poisson_pred.sot_lambda_away),
        }

        for side, lignes in sot_odds.items():
            # 2) λ implicite du marché, moyenné sur les lignes
            #    complètes (over ET under cotés → marge retirable)
            lams_marche = []
            for line, paire in lignes.items():
                if "over" not in paire or "under" not in paire:
                    continue
                probs = novig_probs([paire["over"], paire["under"]])
                if not probs:
                    continue
                lam = PoissonPredictor.fit_lambda_from_over(probs[0], line)
                if lam:
                    lams_marche.append(lam)

            lam_proxy = proxy.get(side, 0)
            if lams_marche:
                lam_marche = sum(lams_marche) / len(lams_marche)
                lam_sot = (PoissonConfig.MARKET_WEIGHT * lam_marche
                           + (1 - PoissonConfig.MARKET_WEIGHT) * lam_proxy
                           if lam_proxy > 0 else lam_marche)
            else:
                lam_sot = lam_proxy  # aucune ligne complète : repli

            if lam_sot <= 0:
                continue

            cible = {"home": home_team, "away": away_team,
                     "total": "du match"}[side]

            # 3) Probabilités et candidats
            for line, paire in lignes.items():
                p_over = PoissonPredictor.poisson_over(lam_sot, line)
                line_txt = line.replace("_", ".")
                for direction, cote in paire.items():
                    p = p_over if direction == "over" else 1 - p_over
                    analysis.model_probs[f"SOT_{side.upper()}"][
                        f"{direction}_{line}"] = round(p, 4)
                    candidates.append((
                        f"Tirs cadrés {cible}",
                        f"{'Over' if direction == 'over' else 'Under'} "
                        f"{line_txt}",
                        cote, p, 15))

        implausible_count = 0

        for market, selection, book_odds, model_prob, penalty in candidates:
            # Cotes 1X2 incohérentes → ne pas parier dessus
            if market == "1X2" and odds_1x2_suspect:
                continue

            # Value aberrante = erreur de données, pas une opportunité
            if (book_odds > 1 and model_prob > 0
                    and model_prob * book_odds - 1
                    > ValueBetConfig.MAX_PLAUSIBLE_VALUE):
                implausible_count += 1
                continue

            vb = self._evaluate_selection(
                match_label, market, selection,
                book_odds, model_prob, model_agreement,
                analysis.bookmaker_margin,
                confidence_penalty=penalty,
                min_value=min_value_eff,
                min_confidence=min_conf_eff,
            )
            if vb:
                analysis.value_bets.append(vb)

        # Trier par value décroissante
        analysis.value_bets.sort(
            key=lambda v: v.value_percentage, reverse=True
        )

        analysis.total_value_bets = len(analysis.value_bets)
        analysis.has_value = analysis.total_value_bets > 0

        # ── Confiance globale ──
        analysis.analysis_confidence = round(
            40 + 40 * model_agreement + min(poisson_pred.confidence, 90) * 0.2,
            1
        )
        analysis.analysis_confidence = min(analysis.analysis_confidence, 95)

        # ── Avertissements ──
        warnings = []
        if odds_1x2_suspect:
            warnings.append(
                f"🚨 Cotes 1X2 incohérentes (marge {analysis.bookmaker_margin:.1f}%) "
                "— probable erreur de lecture de la capture, marché 1X2 ignoré. "
                "Vérifiez les cotes ou utilisez la saisie manuelle."
            )
        if implausible_count:
            warnings.append(
                f"🚨 {implausible_count} value(s) aberrante(s) (> "
                f"{ValueBetConfig.MAX_PLAUSIBLE_VALUE*100:.0f}%) ignorée(s) — "
                "cote probablement mal lue ou mauvais marché"
            )
        if analysis.bookmaker_margin > 8:
            warnings.append(
                f"⚠️ Marge bookmaker élevée ({analysis.bookmaker_margin:.1f}%) "
                "— la value est plus difficile à trouver"
            )
        if model_agreement < 0.85:
            warnings.append(
                "⚠️ Les modèles Poisson et Elo divergent — prudence"
            )
        analysis.data_warning = " | ".join(warnings)

        return analysis

    # ─── ÉVALUATION D'UNE SÉLECTION ─────────────────

    def _evaluate_selection(self, match_label: str, market: str,
                             selection: str, book_odds: float,
                             model_prob: float, model_agreement: float,
                             margin: float,
                             confidence_penalty: float = 0.0,
                             min_value: float = None,
                             min_confidence: float = None
                             ) -> Optional[ValueBet]:
        """
        Évalue une sélection et retourne un ValueBet si les seuils passent.

        confidence_penalty : points de confiance retirés pour les
        marchés modélisés par approximation (mi-temps : 5,
        tirs cadrés : 15).
        """

        if book_odds <= 1 or model_prob <= 0:
            return None

        implied_prob = 1 / book_odds
        value = model_prob * book_odds - 1
        edge = model_prob - implied_prob

        # ── Filtres ──
        # Seuil progressif selon la cote (biais favori-outsider :
        # les grosses cotes exigent plus de value pour être crédibles)
        base_threshold = (min_value if min_value is not None
                          else ValueBetConfig.MIN_VALUE_THRESHOLD)
        multiplier = ValueBetConfig.VALUE_THRESHOLD_MULTIPLIERS[-1][1]
        for odds_cap, mult in ValueBetConfig.VALUE_THRESHOLD_MULTIPLIERS:
            if book_odds < odds_cap:
                multiplier = mult
                break

        if value < base_threshold * multiplier:
            return None
        if not (ValueBetConfig.MIN_ODDS <= book_odds <= ValueBetConfig.MAX_ODDS):
            return None
        if model_prob < ValueBetConfig.MIN_MODEL_PROBABILITY:
            return None

        # ── Score de confiance ──
        # accord des modèles + taille de l'edge + prob raisonnable
        confidence = (
            40 * model_agreement                   # accord des modèles
            + min(value * 200, 30)                 # taille de la value
            + min(model_prob * 50, 25)             # probabilité de l'issue
        )
        # Pénalité si marge élevée (cotes molles, moins fiables)
        if margin > 8:
            confidence -= 8

        # Pénalité des marchés modélisés par approximation
        confidence -= confidence_penalty

        confidence = max(0, min(100, confidence))

        min_conf = (min_confidence if min_confidence is not None
                    else ValueBetConfig.MIN_CONFIDENCE_SCORE)
        if confidence < min_conf:
            return None

        # ── Rating étoiles ──
        rating = "⭐"
        for threshold, stars in ValueBetConfig.RATING_THRESHOLDS:
            if value >= threshold:
                rating = stars
                break

        # ── Kelly (fraction, % du bankroll) ──
        b = book_odds - 1
        kelly_full = (b * model_prob - (1 - model_prob)) / b
        kelly_pct = max(0.0, kelly_full * KellyConfig.KELLY_FRACTION * 100)
        kelly_pct = min(kelly_pct, KellyConfig.MAX_STAKE_PERCENTAGE)

        return ValueBet(
            match=match_label,
            market=market,
            selection=selection,
            bookmaker_odds=book_odds,
            fair_odds=round(1 / model_prob, 2),
            model_probability=round(model_prob, 4),
            implied_probability=round(implied_prob, 4),
            value_percentage=round(value, 4),
            edge=round(edge, 4),
            confidence_score=round(confidence, 1),
            value_rating=rating,
            kelly_stake=round(kelly_pct, 2),
            recommended_stake=0.0,  # rempli par KellyStakeCalculator
        )
