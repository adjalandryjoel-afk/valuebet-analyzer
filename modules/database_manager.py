"""
═══════════════════════════════════════════════════════
 MODULE DATABASE — Gestion de la base de données
 SQLite pour stocker toutes les données du logiciel
═══════════════════════════════════════════════════════
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

from config import Paths


class DatabaseManager:
    """
    Gère le stockage persistant de toutes les données :
    - Matchs analysés
    - Value bets détectés
    - Résultats des paris
    - Ratings Elo historiques
    - Profils xG des équipes
    - Historique du bankroll
    """

    DB_PATH = os.path.join(Paths.DATA_DIR, "valuebet.db")

    def __init__(self):
        os.makedirs(Paths.DATA_DIR, exist_ok=True)
        self._create_tables()

    @contextmanager
    def _get_connection(self):
        """Context manager pour les connexions SQLite."""

        conn = sqlite3.connect(self.DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        """Crée toutes les tables nécessaires."""

        with self._get_connection() as conn:

            # ── Table des matchs analysés ──
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    competition TEXT,
                    match_date TEXT,
                    match_time TEXT,

                    -- Cotes Betclic
                    odds_1 REAL,
                    odds_x REAL,
                    odds_2 REAL,
                    odds_over25 REAL,
                    odds_under25 REAL,
                    odds_btts_yes REAL,
                    odds_btts_no REAL,

                    -- Probabilités du modèle
                    model_prob_1 REAL,
                    model_prob_x REAL,
                    model_prob_2 REAL,
                    model_prob_over25 REAL,
                    model_prob_under25 REAL,
                    model_prob_btts_yes REAL,
                    model_prob_btts_no REAL,

                    -- Lambdas Poisson
                    lambda_home REAL,
                    lambda_away REAL,

                    -- Elo
                    elo_home REAL,
                    elo_away REAL,

                    -- xG
                    xg_home REAL,
                    xg_away REAL,

                    -- Prédiction
                    predicted_score TEXT,
                    predicted_result TEXT,
                    confidence REAL,
                    data_quality TEXT,

                    -- Résultat réel
                    actual_score TEXT,
                    actual_result TEXT,

                    -- Métadonnées
                    analysis_date TEXT DEFAULT (datetime('now')),
                    source_images TEXT,
                    bookmaker_margin REAL,

                    UNIQUE(home_team, away_team, match_date)
                )
            """)

            # ── Table des value bets ──
            conn.execute("""
                CREATE TABLE IF NOT EXISTS value_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER NOT NULL,

                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,

                    bookmaker_odds REAL NOT NULL,
                    fair_odds REAL,
                    model_probability REAL NOT NULL,
                    implied_probability REAL,

                    value_percentage REAL NOT NULL,
                    edge REAL,
                    confidence_score REAL,
                    value_rating TEXT,

                    kelly_stake REAL,
                    recommended_stake REAL,

                    -- Résultat
                    result TEXT,  -- 'win', 'loss', 'void', NULL
                    profit REAL DEFAULT 0,

                    created_at TEXT DEFAULT (datetime('now')),

                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)

            # ── Table du bankroll ──
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bankroll_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    bankroll_amount REAL NOT NULL,
                    daily_profit REAL DEFAULT 0,
                    bets_placed INTEGER DEFAULT 0,
                    bets_won INTEGER DEFAULT 0,
                    bets_lost INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)

            # ── Table des ratings Elo ──
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name TEXT NOT NULL,
                    rating REAL NOT NULL,
                    date TEXT DEFAULT (datetime('now')),
                    match_id INTEGER,

                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)

            # ── Table des profils xG ──
            conn.execute("""
                CREATE TABLE IF NOT EXISTS xg_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name TEXT NOT NULL,
                    league TEXT NOT NULL,
                    season TEXT,

                    avg_xg_for REAL,
                    avg_xg_against REAL,
                    total_xg_for REAL,
                    total_xg_against REAL,
                    xg_difference REAL,
                    goals_minus_xg REAL,

                    home_avg_xg_for REAL,
                    home_avg_xg_against REAL,
                    away_avg_xg_for REAL,
                    away_avg_xg_against REAL,

                    recent_avg_xg_for REAL,
                    recent_avg_xg_against REAL,

                    matches_played INTEGER,
                    last_updated TEXT DEFAULT (datetime('now')),
                    source TEXT,

                    UNIQUE(team_name, league, season)
                )
            """)

            # ── Index pour la performance ──
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_date
                ON matches(match_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_teams
                ON matches(home_team, away_team)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_valuebets_match
                ON value_bets(match_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_elo_team
                ON elo_history(team_name)
            """)

    # ─── OPÉRATIONS SUR LES MATCHS ──────────────────

    def save_match_analysis(self, analysis_data: Dict) -> int:
        """Sauvegarde l'analyse complète d'un match. Retourne l'ID."""

        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO matches (
                    home_team, away_team, competition, match_date,
                    odds_1, odds_x, odds_2,
                    odds_over25, odds_under25,
                    odds_btts_yes, odds_btts_no,
                    model_prob_1, model_prob_x, model_prob_2,
                    model_prob_over25, model_prob_under25,
                    model_prob_btts_yes, model_prob_btts_no,
                    lambda_home, lambda_away,
                    elo_home, elo_away,
                    xg_home, xg_away,
                    predicted_score, predicted_result,
                    confidence, data_quality,
                    bookmaker_margin, source_images
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?
                )
            """, (
                analysis_data.get("home_team"),
                analysis_data.get("away_team"),
                analysis_data.get("competition"),
                analysis_data.get("match_date"),
                analysis_data.get("odds_1"),
                analysis_data.get("odds_x"),
                analysis_data.get("odds_2"),
                analysis_data.get("odds_over25"),
                analysis_data.get("odds_under25"),
                analysis_data.get("odds_btts_yes"),
                analysis_data.get("odds_btts_no"),
                analysis_data.get("model_prob_1"),
                analysis_data.get("model_prob_x"),
                analysis_data.get("model_prob_2"),
                analysis_data.get("model_prob_over25"),
                analysis_data.get("model_prob_under25"),
                analysis_data.get("model_prob_btts_yes"),
                analysis_data.get("model_prob_btts_no"),
                analysis_data.get("lambda_home"),
                analysis_data.get("lambda_away"),
                analysis_data.get("elo_home"),
                analysis_data.get("elo_away"),
                analysis_data.get("xg_home"),
                analysis_data.get("xg_away"),
                analysis_data.get("predicted_score"),
                analysis_data.get("predicted_result"),
                analysis_data.get("confidence"),
                analysis_data.get("data_quality"),
                analysis_data.get("bookmaker_margin"),
                analysis_data.get("source_images"),
            ))

            return cursor.lastrowid

    def save_value_bet(self, match_id: int, vb_data: Dict) -> int:
        """Sauvegarde un value bet détecté."""

        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO value_bets (
                    match_id, market, selection,
                    bookmaker_odds, fair_odds,
                    model_probability, implied_probability,
                    value_percentage, edge,
                    confidence_score, value_rating,
                    kelly_stake, recommended_stake
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                vb_data.get("market"),
                vb_data.get("selection"),
                vb_data.get("bookmaker_odds"),
                vb_data.get("fair_odds"),
                vb_data.get("model_probability"),
                vb_data.get("implied_probability"),
                vb_data.get("value_percentage"),
                vb_data.get("edge"),
                vb_data.get("confidence_score"),
                vb_data.get("value_rating"),
                vb_data.get("kelly_stake"),
                vb_data.get("recommended_stake"),
            ))

            return cursor.lastrowid

    def update_bet_result(self, bet_id: int, result: str, profit: float):
        """Met à jour le résultat d'un value bet."""

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE value_bets
                SET result = ?, profit = ?
                WHERE id = ?
            """, (result, profit, bet_id))

    def update_match_result(self, match_id: int, actual_score: str,
                            actual_result: str):
        """Met à jour le résultat réel d'un match."""

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE matches
                SET actual_score = ?, actual_result = ?
                WHERE id = ?
            """, (actual_score, actual_result, match_id))

    # ─── REQUÊTES DE CONSULTATION ────────────────────

    def get_pending_bets(self) -> List[Dict]:
        """Récupère les paris en attente de résultat."""

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT vb.*, m.home_team, m.away_team, m.match_date
                FROM value_bets vb
                JOIN matches m ON vb.match_id = m.id
                WHERE vb.result IS NULL
                ORDER BY m.match_date DESC
            """).fetchall()

            return [dict(row) for row in rows]

    def get_performance_stats(self) -> Dict:
        """Récupère les statistiques de performance globales."""

        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                    SUM(recommended_stake) as total_staked,
                    SUM(profit) as total_profit,
                    AVG(value_percentage) as avg_value,
                    AVG(confidence_score) as avg_confidence
                FROM value_bets
                WHERE result IS NOT NULL
            """).fetchone()

            if row:
                stats = dict(row)
                stats["roi"] = (
                    (stats["total_profit"] or 0) /
                    max(stats["total_staked"] or 1, 1) * 100
                )
                stats["win_rate"] = (
                    (stats["wins"] or 0) /
                    max(stats["total_bets"] or 1, 1) * 100
                )
                return stats

            return {}

    def get_team_history(self, team_name: str, limit: int = 20) -> List[Dict]:
        """Récupère l'historique des analyses d'une équipe."""

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM matches
                WHERE home_team = ? OR away_team = ?
                ORDER BY match_date DESC
                LIMIT ?
            """, (team_name, team_name, limit)).fetchall()

            return [dict(row) for row in rows]

    def save_xg_profile(self, profile_data: Dict):
        """Sauvegarde ou met à jour un profil xG."""

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO xg_profiles (
                    team_name, league, season,
                    avg_xg_for, avg_xg_against,
                    total_xg_for, total_xg_against,
                    xg_difference, goals_minus_xg,
                    home_avg_xg_for, home_avg_xg_against,
                    away_avg_xg_for, away_avg_xg_against,
                    recent_avg_xg_for, recent_avg_xg_against,
                    matches_played, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile_data.get("team_name"),
                profile_data.get("league"),
                profile_data.get("season"),
                profile_data.get("avg_xg_for"),
                profile_data.get("avg_xg_against"),
                profile_data.get("total_xg_for"),
                profile_data.get("total_xg_against"),
                profile_data.get("xg_difference"),
                profile_data.get("goals_minus_xg"),
                profile_data.get("home_avg_xg_for"),
                profile_data.get("home_avg_xg_against"),
                profile_data.get("away_avg_xg_for"),
                profile_data.get("away_avg_xg_against"),
                profile_data.get("recent_avg_xg_for"),
                profile_data.get("recent_avg_xg_against"),
                profile_data.get("matches_played"),
                profile_data.get("source", "understat"),
            ))
