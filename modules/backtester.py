"""
═══════════════════════════════════════════════════════
 MODULE BACKTESTER — Validation historique des modèles
 et mesure de la performance des prédictions
═══════════════════════════════════════════════════════
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import Paths, KellyConfig


@dataclass
class BetRecord:
    """Enregistrement d'un pari (pour le backtesting)."""

    date: str
    match: str
    market: str
    selection: str
    odds: float
    stake: float
    model_probability: float
    value_percentage: float

    # Résultat
    result: Optional[str] = None  # "win", "loss", "void"
    profit: float = 0.0

    # Métadonnées
    confidence: float = 0.0
    data_quality: str = ""


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest."""

    # Période
    start_date: str = ""
    end_date: str = ""
    total_days: int = 0

    # Volume
    total_bets: int = 0
    total_matches_analyzed: int = 0

    # Performance
    total_staked: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0            # Return on Investment %
    yield_pct: float = 0.0      # Yield (ROI par pari)

    # Détail W/L
    wins: int = 0
    losses: int = 0
    voids: int = 0
    win_rate: float = 0.0

    # Séries
    max_winning_streak: int = 0
    max_losing_streak: int = 0
    current_streak: int = 0

    # Bankroll
    starting_bankroll: float = 0.0
    ending_bankroll: float = 0.0
    max_bankroll: float = 0.0
    min_bankroll: float = 0.0
    max_drawdown: float = 0.0   # % de perte max depuis le pic

    # Qualité des prédictions
    avg_odds: float = 0.0
    avg_value: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0

    # Closing Line Value (CLV)
    avg_clv: float = 0.0       # Moyenne de la valeur vs cote de clôture
    clv_positive_pct: float = 0.0  # % de paris avec CLV > 0

    # Par marché
    performance_by_market: Dict = field(default_factory=dict)

    # Par fourchette de cotes
    performance_by_odds_range: Dict = field(default_factory=dict)

    # Historique du bankroll
    bankroll_history: List[float] = field(default_factory=list)

    # Tous les paris
    bets: List[BetRecord] = field(default_factory=list)

    # Significativité statistique
    p_value: float = 0.0
    is_significant: bool = False


class Backtester:
    """
    Système de backtesting pour valider les performances du modèle.

    Simule les paris historiques et calcule les métriques de performance.

    Métriques clés :
    1. ROI — Rentabilité globale
    2. Yield — ROI par unité misée
    3. CLV — Closing Line Value (battre les cotes de clôture)
    4. Brier Score — Calibration des probabilités
    5. Max Drawdown — Risque maximum
    6. P-value — Significativité statistique
    """

    def __init__(self, starting_bankroll: float = None):
        self.starting_bankroll = starting_bankroll or KellyConfig.DEFAULT_BANKROLL
        self.bet_log: List[BetRecord] = []
        self._load_history()

    def _load_history(self):
        """Charge l'historique des paris."""

        log_path = Paths.RESULTS_LOG

        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.bet_log = [BetRecord(**b) for b in data.get("bets", [])]

    def save_history(self):
        """Sauvegarde l'historique des paris."""

        os.makedirs(Paths.DATA_DIR, exist_ok=True)

        data = {
            "last_updated": datetime.now().isoformat(),
            "total_bets": len(self.bet_log),
            "bets": [
                {
                    "date": b.date,
                    "match": b.match,
                    "market": b.market,
                    "selection": b.selection,
                    "odds": b.odds,
                    "stake": b.stake,
                    "model_probability": b.model_probability,
                    "value_percentage": b.value_percentage,
                    "result": b.result,
                    "profit": b.profit,
                    "confidence": b.confidence,
                    "data_quality": b.data_quality
                }
                for b in self.bet_log
            ]
        }

        with open(Paths.RESULTS_LOG, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ─── ENREGISTREMENT DES PARIS ────────────────────

    def record_bet(self, bet: BetRecord):
        """Enregistre un nouveau pari."""
        self.bet_log.append(bet)
        self.save_history()

    def update_result(self, match: str, market: str,
                      selection: str, result: str):
        """
        Met à jour le résultat d'un pari.

        Args:
            match: Nom du match
            market: Type de marché
            selection: Sélection
            result: "win", "loss", ou "void"
        """

        for bet in self.bet_log:
            if (bet.match == match and bet.market == market and
                bet.selection == selection and bet.result is None):

                bet.result = result

                if result == "win":
                    bet.profit = bet.stake * (bet.odds - 1)
                elif result == "loss":
                    bet.profit = -bet.stake
                else:
                    bet.profit = 0.0

                break

        self.save_history()

    # ─── ANALYSE DES RÉSULTATS ───────────────────────

    def run_backtest(self, bets: List[BetRecord] = None) -> BacktestResult:
        """
        Exécute le backtest complet sur les paris enregistrés.
        """

        if bets is None:
            bets = [b for b in self.bet_log if b.result is not None]

        if not bets:
            print("  ⚠️ Aucun pari résolu pour le backtest")
            return BacktestResult()

        result = BacktestResult(
            starting_bankroll=self.starting_bankroll,
            total_bets=len(bets),
            bets=bets
        )

        # Dates
        dates = [b.date for b in bets if b.date]
        if dates:
            result.start_date = min(dates)
            result.end_date = max(dates)

        # Compteurs de base
        result.wins = sum(1 for b in bets if b.result == "win")
        result.losses = sum(1 for b in bets if b.result == "loss")
        result.voids = sum(1 for b in bets if b.result == "void")
        result.win_rate = result.wins / max(result.total_bets, 1) * 100

        # Financier
        result.total_staked = sum(b.stake for b in bets)
        result.total_profit = sum(b.profit for b in bets)
        result.roi = (
            (result.total_profit / max(result.total_staked, 1)) * 100
        )
        result.yield_pct = result.total_profit / max(result.total_bets, 1)

        # Moyennes
        result.avg_odds = np.mean([b.odds for b in bets])
        result.avg_value = np.mean([b.value_percentage for b in bets])

        # Bankroll évolution
        bankroll = self.starting_bankroll
        max_bankroll = bankroll
        min_bankroll = bankroll
        result.bankroll_history = [bankroll]

        for bet in bets:
            bankroll += bet.profit
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)
            result.bankroll_history.append(bankroll)

        result.ending_bankroll = bankroll
        result.max_bankroll = max_bankroll
        result.min_bankroll = min_bankroll

        # Max Drawdown
        if max_bankroll > 0:
            result.max_drawdown = (
                (max_bankroll - min_bankroll) / max_bankroll * 100
            )

        # Séries
        result.max_winning_streak, result.max_losing_streak = (
            self._calculate_streaks(bets)
        )

        # Performance par marché
        result.performance_by_market = self._performance_by_category(
            bets, lambda b: b.market
        )

        # Performance par fourchette de cotes
        def odds_range(bet):
            if bet.odds < 1.5:
                return "1.00-1.50"
            elif bet.odds < 2.0:
                return "1.50-2.00"
            elif bet.odds < 3.0:
                return "2.00-3.00"
            elif bet.odds < 5.0:
                return "3.00-5.00"
            else:
                return "5.00+"

        result.performance_by_odds_range = self._performance_by_category(
            bets, odds_range
        )

        # Brier Score
        result.brier_score = self._calculate_brier_score(bets)

        # Significativité statistique (test z)
        result.p_value = self._calculate_p_value(bets)
        result.is_significant = result.p_value < 0.05

        return result

    def _calculate_streaks(self, bets: List[BetRecord]) -> Tuple[int, int]:
        """Calcule les plus longues séries gagnantes et perdantes."""

        max_win = 0
        max_loss = 0
        current_win = 0
        current_loss = 0

        for bet in bets:
            if bet.result == "win":
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            elif bet.result == "loss":
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)
            else:
                current_win = 0
                current_loss = 0

        return max_win, max_loss

    def _performance_by_category(self, bets: List[BetRecord],
                                  categorizer) -> Dict:
        """Calcule la performance par catégorie."""

        categories = {}

        for bet in bets:
            cat = categorizer(bet)
            if cat not in categories:
                categories[cat] = {"bets": 0, "wins": 0, "staked": 0, "profit": 0}

            categories[cat]["bets"] += 1
            categories[cat]["staked"] += bet.stake
            categories[cat]["profit"] += bet.profit

            if bet.result == "win":
                categories[cat]["wins"] += 1

        # Calculer ROI par catégorie
        for cat in categories:
            d = categories[cat]
            d["win_rate"] = d["wins"] / max(d["bets"], 1) * 100
            d["roi"] = d["profit"] / max(d["staked"], 1) * 100

        return categories

    def _calculate_brier_score(self, bets: List[BetRecord]) -> float:
        """
        Calcule le Brier Score (mesure de calibration).

        Brier Score = (1/N) × Σ(probabilité - résultat)²

        Plus c'est bas, mieux c'est. < 0.25 = bon.
        """

        if not bets:
            return 0.0

        scores = []
        for bet in bets:
            actual = 1.0 if bet.result == "win" else 0.0
            predicted = bet.model_probability
            scores.append((predicted - actual) ** 2)

        return np.mean(scores)

    def _calculate_p_value(self, bets: List[BetRecord]) -> float:
        """
        Calcule la p-value pour déterminer si les résultats
        sont statistiquement significatifs.

        H0 : Le modèle n'a pas d'avantage (ROI = 0)
        H1 : Le modèle a un avantage (ROI > 0)
        """

        if len(bets) < 30:
            return 1.0  # Pas assez de données

        profits = [b.profit / max(b.stake, 1) for b in bets]

        mean = np.mean(profits)
        std = np.std(profits)
        n = len(profits)

        if std == 0:
            return 1.0

        # Test z unilatéral
        z_score = mean / (std / np.sqrt(n))

        from scipy import stats
        p_value = 1 - stats.norm.cdf(z_score)

        return p_value

    # ─── RAPPORT DE BACKTEST ─────────────────────────

    def print_backtest_report(self, result: BacktestResult):
        """Affiche le rapport de backtest formaté."""

        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()

        console.print(f"\n{'▓'*70}", style="bold yellow")
        console.print("  📊 RAPPORT DE BACKTESTING", style="bold yellow")
        console.print(f"{'▓'*70}\n", style="bold yellow")

        # Stats globales
        roi_style = "green" if result.roi > 0 else "red"

        console.print(f"  📅 Période : {result.start_date} → {result.end_date}")
        console.print(f"  🎰 Total paris : {result.total_bets}")
        console.print(f"  ✅ Gagnés : {result.wins} ({result.win_rate:.1f}%)")
        console.print(f"  ❌ Perdus : {result.losses}")
        console.print(f"  💰 Total misé : {result.total_staked:,.0f} FCFA")
        console.print(
            f"  📈 Profit : [{roi_style}]{result.total_profit:+,.0f} FCFA[/]"
        )
        console.print(
            f"  🎯 ROI : [{roi_style}]{result.roi:+.2f}%[/]"
        )
        console.print(
            f"  📉 Max Drawdown : {result.max_drawdown:.1f}%"
        )
        console.print(
            f"  🏦 Bankroll final : {result.ending_bankroll:,.0f} FCFA"
        )

        # Brier Score
        brier_style = "green" if result.brier_score < 0.25 else "yellow"
        console.print(
            f"  🎯 Brier Score : [{brier_style}]{result.brier_score:.4f}[/]"
        )

        # Significativité
        sig_style = "green" if result.is_significant else "red"
        console.print(
            f"  📊 P-value : [{sig_style}]{result.p_value:.4f}[/] "
            f"({'Significatif ✅' if result.is_significant else 'Non significatif ❌'})"
        )

        # Séries
        console.print(
            f"  🏆 Meilleure série : {result.max_winning_streak} gagnants"
        )
        console.print(
            f"  💀 Pire série : {result.max_losing_streak} perdants"
        )

        # Performance par marché
        if result.performance_by_market:
            market_table = Table(
                title="\n📊 Performance par marché",
                box=box.ROUNDED
            )
            market_table.add_column("Marché", style="cyan")
            market_table.add_column("Paris", justify="center")
            market_table.add_column("Win%", justify="center")
            market_table.add_column("ROI", justify="center")

            for market, stats in sorted(
                result.performance_by_market.items(),
                key=lambda x: x[1]["roi"], reverse=True
            ):
                roi_s = "green" if stats["roi"] > 0 else "red"
                market_table.add_row(
                    market[:25],
                    str(stats["bets"]),
                    f"{stats['win_rate']:.1f}%",
                    f"[{roi_s}]{stats['roi']:+.1f}%[/]"
                )

            console.print(market_table)

        # Performance par fourchette de cotes
        if result.performance_by_odds_range:
            odds_table = Table(
                title="\n📊 Performance par fourchette de cotes",
                box=box.ROUNDED
            )
            odds_table.add_column("Cotes", style="cyan")
            odds_table.add_column("Paris", justify="center")
            odds_table.add_column("Win%", justify="center")
            odds_table.add_column("ROI", justify="center")

            for odds_range, stats in sorted(
                result.performance_by_odds_range.items()
            ):
                roi_s = "green" if stats["roi"] > 0 else "red"
                odds_table.add_row(
                    odds_range,
                    str(stats["bets"]),
                    f"{stats['win_rate']:.1f}%",
                    f"[{roi_s}]{stats['roi']:+.1f}%[/]"
                )

            console.print(odds_table)
