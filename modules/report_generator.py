"""
═══════════════════════════════════════════════════════
 MODULE REPORT — Rapports console (rich) et export JSON
═══════════════════════════════════════════════════════
"""

import os
import json
from datetime import datetime
from typing import List

from config import Paths
from modules.value_detector import MatchAnalysis
from modules.kelly_criterion import StakeRecommendation

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class ReportGenerator:
    """Affichage console des analyses + export JSON."""

    def __init__(self, bankroll: float = 100_000):
        self.bankroll = bankroll
        self.console = Console() if HAS_RICH else None

    def _print(self, text: str, style: str = None):
        if self.console:
            self.console.print(text, style=style)
        else:
            print(text)

    # ─── EN-TÊTE ────────────────────────────────────

    def print_header(self):
        self._print(f"\n{'═'*70}", "bold cyan")
        self._print(
            "  ⚽ VALUE BET ANALYZER v2.0 — BETCLIC CÔTE D'IVOIRE",
            "bold cyan"
        )
        self._print(f"  💰 Bankroll : {self.bankroll:,.0f} FCFA", "cyan")
        self._print(
            f"  📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}", "cyan"
        )
        self._print(f"{'═'*70}", "bold cyan")

    # ─── RAPPORT PAR MATCH ──────────────────────────

    def print_match_report(self, analysis: MatchAnalysis,
                           stakes: List[StakeRecommendation] = None):
        """Affiche le rapport détaillé d'un match."""

        icon = "🟢" if analysis.has_value else "🔴"

        self._print(
            f"\n{icon} {analysis.home_team} vs {analysis.away_team}"
            + (f" — {analysis.competition}" if analysis.competition else ""),
            "bold white"
        )

        # Probabilités 1X2
        probs = analysis.model_probs.get("1X2", {})

        if HAS_RICH:
            table = Table(box=box.SIMPLE, show_header=True)
            table.add_column("Sélection", style="cyan")
            table.add_column("Prob. modèle", justify="center")
            table.add_column("Prob. Betclic", justify="center")
            table.add_column("Cote", justify="center")
            table.add_column("Value", justify="center")

            for key, label in (("1", f"1 — {analysis.home_team}"),
                               ("X", "X — Nul"),
                               ("2", f"2 — {analysis.away_team}")):
                mp = probs.get(key, 0)
                bo = float(analysis.odds.get(key, 0) or 0)
                implied = f"{100/bo:.1f}%" if bo > 1 else "—"
                if bo > 1 and mp > 0:
                    v = (bo * mp - 1) * 100
                    v_style = "green" if v > 0 else "red"
                    value_str = f"[{v_style}]{v:+.1f}%[/]"
                else:
                    value_str = "—"

                table.add_row(
                    label, f"{mp*100:.1f}%", implied,
                    f"{bo:.2f}" if bo > 1 else "—", value_str
                )

            self.console.print(table)
        else:
            for key in ("1", "X", "2"):
                mp = probs.get(key, 0)
                bo = float(analysis.odds.get(key, 0) or 0)
                print(f"    {key}: modèle {mp*100:.1f}% | cote {bo:.2f}")

        self._print(
            f"  🎯 Score prédit : {analysis.predicted_score} | "
            f"{analysis.most_likely_result} | "
            f"Marge Betclic : {analysis.bookmaker_margin:.1f}% | "
            f"Confiance : {analysis.analysis_confidence:.0f}/100"
        )

        # Value bets
        if analysis.value_bets:
            self._print(
                f"\n  ✅ {analysis.total_value_bets} VALUE BET(S) :",
                "bold green"
            )
            for vb in analysis.value_bets:
                self._print(
                    f"    ▸ {vb.market} → {vb.selection} @ {vb.bookmaker_odds:.2f} "
                    f"| value +{vb.value_percentage*100:.1f}% {vb.value_rating} "
                    f"| confiance {vb.confidence_score:.0f}/100 "
                    f"| mise {vb.recommended_stake:,.0f} FCFA "
                    f"({vb.kelly_stake:.1f}%)",
                    "green"
                )
        else:
            self._print("  ❌ Pas de value bet sur ce match", "red")

        if analysis.data_warning:
            self._print(f"  {analysis.data_warning}", "yellow")

    # ─── RÉSUMÉ GLOBAL ──────────────────────────────

    def print_summary(self, analyses: List[MatchAnalysis],
                      stakes: List[StakeRecommendation] = None):
        """Affiche le résumé de la session d'analyse."""

        total_matches = len(analyses)
        with_value = sum(1 for a in analyses if a.has_value)
        total_vb = sum(a.total_value_bets for a in analyses)
        total_stake = sum(
            vb.recommended_stake for a in analyses for vb in a.value_bets
        )
        potential = sum(
            vb.recommended_stake * (vb.bookmaker_odds - 1)
            for a in analyses for vb in a.value_bets
        )

        self._print(f"\n{'═'*70}", "bold cyan")
        self._print("  📋 RÉSUMÉ DE LA SESSION", "bold cyan")
        self._print(f"{'═'*70}", "bold cyan")
        self._print(f"  ⚽ Matchs analysés      : {total_matches}")
        self._print(f"  ✅ Matchs avec value    : {with_value}")
        self._print(f"  🎰 Value bets détectés  : {total_vb}")
        self._print(f"  💰 Investissement total : {total_stake:,.0f} FCFA "
                    f"({total_stake/max(self.bankroll,1)*100:.1f}% du bankroll)")
        self._print(f"  📈 Gain potentiel net   : {potential:,.0f} FCFA")

        # Meilleurs paris
        all_vbs = [vb for a in analyses for vb in a.value_bets]
        all_vbs.sort(key=lambda v: v.value_percentage, reverse=True)

        if all_vbs:
            self._print("\n  🏆 TOP VALUE BETS :", "bold yellow")
            for vb in all_vbs[:5]:
                self._print(
                    f"    {vb.value_rating} {vb.match} | {vb.market} → "
                    f"{vb.selection} @ {vb.bookmaker_odds:.2f} "
                    f"(+{vb.value_percentage*100:.1f}%)",
                    "yellow"
                )

    # ─── EXPORT JSON ────────────────────────────────

    def export_to_json(self, analyses: List[MatchAnalysis]) -> str:
        """Exporte toutes les analyses en JSON dans reports/."""

        os.makedirs(Paths.REPORTS_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(Paths.REPORTS_DIR, f"analyse_{timestamp}.json")

        data = {
            "generated_at": datetime.now().isoformat(),
            "bankroll": self.bankroll,
            "total_matches": len(analyses),
            "total_value_bets": sum(a.total_value_bets for a in analyses),
            "matches": [
                {
                    "home_team": a.home_team,
                    "away_team": a.away_team,
                    "competition": a.competition,
                    "odds": a.odds,
                    "model_probs": a.model_probs,
                    "predicted_score": a.predicted_score,
                    "most_likely_result": a.most_likely_result,
                    "bookmaker_margin": a.bookmaker_margin,
                    "analysis_confidence": a.analysis_confidence,
                    "value_bets": [
                        {
                            "market": vb.market,
                            "selection": vb.selection,
                            "bookmaker_odds": vb.bookmaker_odds,
                            "fair_odds": vb.fair_odds,
                            "model_probability": vb.model_probability,
                            "value_percentage": vb.value_percentage,
                            "edge": vb.edge,
                            "confidence_score": vb.confidence_score,
                            "value_rating": vb.value_rating,
                            "kelly_stake": vb.kelly_stake,
                            "recommended_stake": vb.recommended_stake,
                        }
                        for vb in a.value_bets
                    ],
                }
                for a in analyses
            ],
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._print(f"\n  💾 Rapport exporté : {path}", "cyan")
        return path
