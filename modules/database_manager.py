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

            # ── Pierres tombales du miroir cloud ──
            # bet_keys supprimés localement par supersede : ne
            # jamais les ré-importer ni les re-pousser vers le cloud
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cloud_tombstones (
                    bet_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # ── Migrations idempotentes (CLV tracking) ──
            # Ajoute les colonnes de clôture aux bases existantes ;
            # sqlite3.OperationalError "duplicate column" = déjà migré.
            for ddl in (
                "ALTER TABLE value_bets ADD COLUMN closing_odds REAL",
                "ALTER TABLE value_bets ADD COLUMN clv_pct REAL",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass

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

    # ─── MIROIR CLOUD (Supabase) ─────────────────────
    # Le SQLite local reste la source de vérité de l'appareil ;
    # Supabase est un miroir permanent partagé entre le PC et le
    # cloud Streamlit. Chaque écriture importante est poussée au
    # fil de l'eau ; hydrate_from_cloud() reconstitue le local au
    # démarrage. Tout est silencieux en cas d'échec réseau.

    @property
    def cloud(self):
        if not hasattr(self, "_cloud_checked"):
            self._cloud_checked = True
            self._cloud = None
            try:
                from modules.cloud_store import get_cloud_store
                self._cloud = get_cloud_store()
            except Exception:
                pass
        return self._cloud

    @staticmethod
    def _match_key(row: Dict) -> str:
        """Clé naturelle stable d'un match, identique sur tous les
        appareils : équipes + date du match (ou jour d'analyse)."""
        date = (row.get("match_date")
                or str(row.get("analysis_date") or "")[:10])
        return (f"{str(row.get('home_team') or '').strip()}|"
                f"{str(row.get('away_team') or '').strip()}|{date}")

    @classmethod
    def _bet_key(cls, match_row: Dict, bet_row: Dict) -> str:
        # created_at (fixé par SQLite à l'insertion, transporté tel
        # quel dans le payload cloud) discrimine chaque analyse : un
        # nouveau pari ne peut jamais retomber sur la clé d'un pari
        # antérieur déjà réglé et écraser son résultat.
        return (f"{cls._match_key(match_row)}|"
                f"{bet_row.get('market')}|{bet_row.get('selection')}|"
                f"{str(bet_row.get('created_at') or '')}")

    def _row_dict(self, table: str, row_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", (row_id,)
            ).fetchone()
        return dict(row) if row else None

    def _cloud_push_match(self, match_id: int):
        if not self.cloud or not match_id:
            return
        try:
            m = self._row_dict("matches", match_id)
            if m:
                payload = {k: v for k, v in m.items() if k != "id"}
                self.cloud.upsert_match(self._match_key(m), payload)
        except Exception:
            pass

    def _cloud_push_bet(self, bet_id: int):
        if not self.cloud or not bet_id:
            return
        try:
            b = self._row_dict("value_bets", bet_id)
            if not b:
                return
            m = self._row_dict("matches", b.get("match_id") or 0)
            if not m:
                return
            bet_key = self._bet_key(m, b)

            # Jamais re-pousser un pari supersédé : il ressusciterait
            # dans le cloud puis sur tous les appareils
            with self._get_connection() as conn:
                mort = conn.execute(
                    "SELECT 1 FROM cloud_tombstones WHERE bet_key = ?",
                    (bet_key,)).fetchone()
            if mort:
                return

            payload = {k: v for k, v in b.items()
                       if k not in ("id", "match_id")}
            self.cloud.upsert_bet(bet_key, self._match_key(m), payload)
        except Exception:
            pass

    def hydrate_from_cloud(self) -> Dict:
        """
        Reconstitue/synchronise le SQLite local depuis Supabase :
          • matchs et paris absents localement → insérés ;
          • paris locaux en attente réglés ailleurs → mis à jour
            (résultat, profit, cote de clôture, CLV) ;
          • résultats de matchs enregistrés ailleurs → repris.
        Retourne {"matchs": n, "paris": n, "maj": n}.
        """

        bilan = {"matchs": 0, "paris": 0, "maj": 0}
        if not self.cloud:
            return bilan

        try:
            remote_matches = self.cloud.fetch_matches()
            remote_bets = self.cloud.fetch_bets()
        except Exception:
            return bilan
        if not remote_matches and not remote_bets:
            return bilan

        a_remarquer = []  # pierres tombales à re-poser côté cloud

        try:
            with self._get_connection() as conn:
                # Sérialise avec le robot CLV (autre process) : une
                # seule hydratation écrit à la fois, l'index lu reste
                # cohérent avec les insertions qui suivent
                conn.execute("BEGIN IMMEDIATE")

                mcols = {r[1] for r in conn.execute(
                    "PRAGMA table_info(matches)")}
                bcols = {r[1] for r in conn.execute(
                    "PRAGMA table_info(value_bets)")}
                tombales = {r[0] for r in conn.execute(
                    "SELECT bet_key FROM cloud_tombstones")}

                # ── Index local : clé naturelle → ligne ──
                local_matches = {}
                for row in conn.execute("SELECT * FROM matches"):
                    local_matches[self._match_key(dict(row))] = dict(row)

                key_to_id = {k: v["id"]
                             for k, v in local_matches.items()}

                # ── Matchs distants ──
                for rm in remote_matches:
                    key = rm.get("match_key")
                    payload = rm.get("payload") or {}
                    if not key or not isinstance(payload, dict):
                        continue

                    if key not in key_to_id:
                        data = {k: v for k, v in payload.items()
                                if k in mcols and k != "id"}
                        if not data.get("home_team"):
                            continue
                        cols = ", ".join(data)
                        marks = ", ".join("?" * len(data))
                        cur = conn.execute(
                            f"INSERT INTO matches ({cols}) "
                            f"VALUES ({marks})", list(data.values()))
                        key_to_id[key] = cur.lastrowid
                        bilan["matchs"] += 1
                    else:
                        local = local_matches[key]
                        if (payload.get("actual_result")
                                and not local.get("actual_result")):
                            conn.execute(
                                "UPDATE matches SET actual_score = ?, "
                                "actual_result = ? WHERE id = ?",
                                (payload.get("actual_score"),
                                 payload.get("actual_result"),
                                 local["id"]))
                            bilan["maj"] += 1

                # ── Index local des paris ──
                local_bets = {}
                for row in conn.execute("""
                    SELECT vb.*, m.home_team AS _h, m.away_team AS _a,
                           m.match_date AS _md, m.analysis_date AS _ad
                    FROM value_bets vb
                    JOIN matches m ON m.id = vb.match_id
                """):
                    b = dict(row)
                    mrow = {"home_team": b["_h"], "away_team": b["_a"],
                            "match_date": b["_md"],
                            "analysis_date": b["_ad"]}
                    local_bets[self._bet_key(mrow, b)] = b

                # ── Paris distants ──
                for rb in remote_bets:
                    key = rb.get("bet_key")
                    payload = rb.get("payload") or {}
                    if not key or not isinstance(payload, dict):
                        continue

                    # Colonnes de tête (réglées par le robot CLV ou
                    # un autre appareil) prioritaires sur le payload
                    for col in ("result", "closing_odds", "clv_pct"):
                        if rb.get(col) is not None:
                            payload[col] = rb[col]

                    resultat_distant = payload.get("result")

                    # Pari supprimé ICI par une ré-analyse : jamais
                    # ré-importé. S'il est resté « en attente » côté
                    # cloud (échec réseau passé), re-poser la tombale.
                    if key in tombales:
                        if resultat_distant is None:
                            a_remarquer.append(key)
                        continue

                    # Pari supersédé AILLEURS : purge du pending local
                    if resultat_distant == "superseded":
                        local = local_bets.get(key)
                        if local and not local.get("result"):
                            conn.execute(
                                "DELETE FROM value_bets WHERE id = ?",
                                (local["id"],))
                            bilan["maj"] += 1
                        continue

                    if key not in local_bets:
                        match_id = key_to_id.get(rb.get("match_key"))
                        if not match_id:
                            # Match jamais poussé (échec réseau passé) :
                            # squelette reconstruit depuis la clé
                            # home|away|date pour ne pas perdre le pari
                            parts = str(rb.get("match_key")
                                        or "").split("|")
                            if (len(parts) < 3 or not parts[0]
                                    or not parts[1]):
                                continue
                            cur = conn.execute(
                                "INSERT INTO matches (home_team, "
                                "away_team, match_date) "
                                "VALUES (?, ?, ?)",
                                (parts[0], parts[1],
                                 parts[2] or None))
                            match_id = cur.lastrowid
                            key_to_id[rb.get("match_key")] = match_id
                            bilan["matchs"] += 1

                        data = {k: v for k, v in payload.items()
                                if k in bcols and k != "id"}
                        data["match_id"] = match_id
                        if not data.get("market"):
                            continue
                        cols = ", ".join(data)
                        marks = ", ".join("?" * len(data))
                        conn.execute(
                            f"INSERT INTO value_bets ({cols}) "
                            f"VALUES ({marks})", list(data.values()))
                        bilan["paris"] += 1
                    else:
                        local = local_bets[key]
                        updates = {}
                        if (payload.get("result")
                                and not local.get("result")):
                            updates["result"] = payload["result"]
                            updates["profit"] = payload.get("profit", 0)
                        if (payload.get("closing_odds")
                                and not local.get("closing_odds")):
                            updates["closing_odds"] = \
                                payload["closing_odds"]
                            updates["clv_pct"] = payload.get("clv_pct")
                        if updates:
                            sets = ", ".join(
                                f"{c} = ?" for c in updates)
                            conn.execute(
                                f"UPDATE value_bets SET {sets} "
                                f"WHERE id = ?",
                                list(updates.values()) + [local["id"]])
                            bilan["maj"] += 1
        except Exception as e:
            print(f"   ⚠️ Hydratation cloud interrompue : "
                  f"{type(e).__name__}")
            return bilan

        # Rattrapage réseau : re-pose des tombales côté cloud
        for key in a_remarquer[:20]:
            try:
                self.cloud.mark_superseded_key(key)
            except Exception:
                pass

        return bilan

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
            new_id = cursor.lastrowid

        self._cloud_push_match(new_id)
        return new_id

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
            new_id = cursor.lastrowid

        self._cloud_push_bet(new_id)
        return new_id

    def update_bet_result(self, bet_id: int, result: str, profit: float):
        """Met à jour le résultat d'un value bet."""

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE value_bets
                SET result = ?, profit = ?
                WHERE id = ?
            """, (result, profit, bet_id))

        self._cloud_push_bet(bet_id)

    def update_bet_closing(self, bet_id: int, closing_odds: float,
                           clv_pct: float):
        """
        Enregistre la cote de clôture et le CLV d'un value bet.

        Args:
            bet_id: ID du value bet
            closing_odds: cote de clôture de la sélection
            clv_pct: Closing Line Value en % (cote prise vs cote
                     juste de clôture sans marge)
        """

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE value_bets
                SET closing_odds = ?, clv_pct = ?
                WHERE id = ?
            """, (closing_odds, clv_pct, bet_id))

        self._cloud_push_bet(bet_id)

    def update_match_result(self, match_id: int, actual_score: str,
                            actual_result: str):
        """Met à jour le résultat réel d'un match."""

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE matches
                SET actual_score = ?, actual_result = ?
                WHERE id = ?
            """, (actual_score, actual_result, match_id))

        self._cloud_push_match(match_id)

    # ─── REQUÊTES DE CONSULTATION ────────────────────

    def supersede_pending_bets(self, home_team: str, away_team: str):
        """
        Supprime les paris EN ATTENTE des analyses précédentes du même
        match : une ré-analyse (cotes fraîches) remplace les anciens
        signaux au lieu de les dupliquer dans le suivi. Les paris déjà
        résolus (gagné/perdu/non joué) sont conservés.
        """

        with self._get_connection() as conn:
            # Pierres tombales : mémorise les bet_keys supprimés pour
            # ne jamais les ré-importer ni les re-pousser (la même
            # transaction que la suppression → jamais d'écart)
            rows = conn.execute("""
                SELECT vb.*, m.home_team AS _h, m.away_team AS _a,
                       m.match_date AS _md, m.analysis_date AS _ad
                FROM value_bets vb
                JOIN matches m ON m.id = vb.match_id
                WHERE vb.result IS NULL
                  AND m.home_team = ? AND m.away_team = ?
            """, (home_team, away_team)).fetchall()

            for r in rows:
                b = dict(r)
                mrow = {"home_team": b["_h"], "away_team": b["_a"],
                        "match_date": b["_md"],
                        "analysis_date": b["_ad"]}
                conn.execute(
                    "INSERT OR IGNORE INTO cloud_tombstones "
                    "(bet_key) VALUES (?)", (self._bet_key(mrow, b),))

            conn.execute(
                "DELETE FROM cloud_tombstones "
                "WHERE created_at < datetime('now', '-30 days')")

            conn.execute("""
                DELETE FROM value_bets
                WHERE result IS NULL
                  AND match_id IN (
                      SELECT id FROM matches
                      WHERE home_team = ? AND away_team = ?
                  )
            """, (home_team, away_team))

        # Propagation cloud PAR AFFICHE (couvre aussi les analyses
        # d'un autre jour ou d'un autre appareil, que ce SQLite ne
        # connaît pas). Marquage, jamais suppression : une ligne
        # marquée ne peut pas être ressuscitée par un push retardataire.
        if self.cloud:
            try:
                self.cloud.mark_superseded(
                    home_team.strip(), away_team.strip())
            except Exception:
                pass

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
                WHERE result IS NOT NULL AND result != 'void'
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

    def get_clv_stats(self) -> Dict:
        """
        Statistiques de Closing Line Value sur les paris dont la
        cote de clôture a été capturée (clv_pct non NULL).

        Returns:
            {"n_avec_clv": int, "clv_moyen_pct": float,
             "clv_positif_pct": float (% de paris à CLV > 0)}
        """

        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS n_avec_clv,
                    AVG(clv_pct) AS clv_moyen_pct,
                    AVG(CASE WHEN clv_pct > 0 THEN 100.0 ELSE 0.0 END)
                        AS clv_positif_pct
                FROM value_bets
                WHERE clv_pct IS NOT NULL
            """).fetchone()

            n = row["n_avec_clv"] if row else 0
            return {
                "n_avec_clv": n or 0,
                "clv_moyen_pct": (row["clv_moyen_pct"] or 0.0) if n else 0.0,
                "clv_positif_pct": (row["clv_positif_pct"] or 0.0) if n else 0.0,
            }

    def get_analysis_history(self, limit: int = 200,
                             search: str = "") -> List[Dict]:
        """
        Historique des analyses, les plus récentes d'abord, avec le
        nombre de value bets et la meilleure value de chaque match.
        """

        with self._get_connection() as conn:
            query = """
                SELECT
                    m.id, m.home_team, m.away_team, m.competition,
                    m.analysis_date, m.predicted_score, m.predicted_result,
                    m.confidence, m.bookmaker_margin,
                    m.odds_1, m.odds_x, m.odds_2,
                    COUNT(vb.id) AS n_value_bets,
                    MAX(vb.value_percentage) AS best_value,
                    SUM(CASE WHEN vb.result IS NOT NULL
                        THEN 1 ELSE 0 END) AS n_resolved
                FROM matches m
                LEFT JOIN value_bets vb ON vb.match_id = m.id
            """
            params: list = []

            if search:
                query += """
                    WHERE m.home_team LIKE ? OR m.away_team LIKE ?
                       OR m.competition LIKE ?
                """
                needle = f"%{search}%"
                params = [needle, needle, needle]

            query += """
                GROUP BY m.id
                ORDER BY m.analysis_date DESC, m.id DESC
                LIMIT ?
            """
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_bets_for_simulation(self) -> List[Dict]:
        """
        Tous les paris enregistrés avec une mise (résolus ou non) —
        matière première du simulateur Monte Carlo.
        """

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT bookmaker_odds, recommended_stake, model_probability
                FROM value_bets
                WHERE recommended_stake > 0
                  AND model_probability > 0
                  AND (result IS NULL OR result != 'void')
                ORDER BY created_at ASC, id ASC
            """).fetchall()

            return [dict(row) for row in rows]

    def get_matches_awaiting_result(self) -> List[Dict]:
        """
        Matchs analysés ayant encore des paris en attente —
        candidats à la saisie/récupération du score final.
        """

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT m.id, m.home_team, m.away_team, m.competition,
                       m.analysis_date, COUNT(vb.id) AS n_pending
                FROM matches m
                JOIN value_bets vb ON vb.match_id = m.id
                WHERE vb.result IS NULL
                GROUP BY m.id
                ORDER BY m.analysis_date DESC
            """).fetchall()

            return [dict(row) for row in rows]

    def get_calibration_rows(self) -> List[Dict]:
        """
        Analyses dont le résultat réel est connu : matière première
        de la calibration continue (probabilités 1X2 du modèle vs
        issues réelles).
        """

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT model_prob_1, model_prob_x, model_prob_2,
                       odds_1, odds_x, odds_2, actual_result
                FROM matches
                WHERE actual_result IN ('1', 'X', '2')
                  AND model_prob_1 IS NOT NULL
            """).fetchall()

            return [dict(row) for row in rows]

    def get_resolved_bets(self) -> List[Dict]:
        """
        Value bets résolus (hors « non joué »), du plus ancien au plus
        récent — pour les graphiques de profit et de performance.
        """

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT market, selection, bookmaker_odds,
                       recommended_stake, result, profit, created_at
                FROM value_bets
                WHERE result IS NOT NULL AND result != 'void'
                ORDER BY created_at ASC, id ASC
            """).fetchall()

            return [dict(row) for row in rows]

    def get_value_bets_for_match(self, match_id: int) -> List[Dict]:
        """Value bets enregistrés pour une analyse donnée."""

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM value_bets
                WHERE match_id = ?
                ORDER BY value_percentage DESC
            """, (match_id,)).fetchall()

            return [dict(row) for row in rows]

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
