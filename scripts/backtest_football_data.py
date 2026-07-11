"""
═══════════════════════════════════════════════════════════════
 BACKTEST — Modèle Poisson réel de l'app sur football-data.co.uk
═══════════════════════════════════════════════════════════════

Rejoue le VRAI modèle de l'app (PoissonPredictor, TeamStats,
MatchContext, novig_probs/Shin) sur 5 saisons × 5 grands
championnats téléchargés depuis football-data.co.uk.

- Stats des équipes reconstruites match par match depuis leurs
  matchs PRÉCÉDENTS uniquement (fenêtre 15, décote temporelle
  exp(-TIME_DECAY_XI × jours)) → TeamStats(data_source="api").
- λ marché via PoissonPredictor._lambdas_from_market (cotes B365
  no-vig Shin), λ stats via _lambdas_from_stats, puis blend
  manuel w×marché + (1-w)×stats — réplique exacte de
  _estimate_lambdas sans muter PoissonConfig.
- Train (2122-2324) : grille MARKET_WEIGHT → meilleur poids
  par log-loss 1X2.
- Test (2425-2526) : Brier/log-loss modèle vs marché no-vig
  (B365 et Pinnacle clôture), calibration par déciles,
  stratégie value quart-Kelly, CLV vs clôture Pinnacle.

Sortie : data/backtest_results.json + résumé console.
Aucun fichier existant n'est modifié.
"""

import os
import sys
import json
import math
import time
from datetime import datetime, timezone

import pandas as pd

# Rendre importable le paquet de l'app (le script vit dans scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import PoissonConfig, ValueBetConfig, KellyConfig, SUPPORTED_LEAGUES  # noqa: E402
from modules.data_collector import TeamStats, MatchContext                        # noqa: E402
from modules.poisson_model import PoissonPredictor                                # noqa: E402
from modules.odds_utils import novig_probs                                        # noqa: E402


# ══════════════════════════════════════════════════════════════
#  PARAMÈTRES DU BACKTEST
# ══════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "football_data")
RESULTS_PATH = os.path.join(DATA_DIR, "backtest_results.json")

# 2021 n'est téléchargée QUE pour la moyenne de buts de la ligue
# de la saison précédant 2122 (pas de matchs backtestés dessus).
SEASONS_ALL = ["2021", "2122", "2223", "2324", "2425", "2526"]
SEASONS_BACKTEST = ["2122", "2223", "2324", "2425", "2526"]
SEASONS_TRAIN = {"2122", "2223", "2324"}
SEASONS_TEST = {"2425", "2526"}

# division football-data → clé SUPPORTED_LEAGUES
DIVISIONS = {
    "E0": "premier_league",
    "SP1": "la_liga",
    "I1": "serie_a",
    "D1": "bundesliga",
    "F1": "ligue1_fr",
}

MARKET_WEIGHT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]

MIN_HISTORY = 5      # matchs d'historique minimum par équipe
WINDOW = 15          # fenêtre de matchs récents
XI = PoissonConfig.TIME_DECAY_XI  # décote temporelle exp(-xi × jours)

BANKROLL = KellyConfig.DEFAULT_BANKROLL           # 100 000, fixe (pas de capitalisation)
KELLY_FRACTION = KellyConfig.KELLY_FRACTION       # quart de Kelly
MAX_STAKE = BANKROLL * KellyConfig.MAX_STAKE_PERCENTAGE / 100  # plafond 2%

USE_COLS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
    "HST", "AST", "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
    "PSCH", "PSCD", "PSCA", "B365>2.5", "B365<2.5",
]


# ══════════════════════════════════════════════════════════════
#  1. TÉLÉCHARGEMENT + CACHE DISQUE
# ══════════════════════════════════════════════════════════════

def download_all() -> dict:
    """Télécharge (ou lit depuis le cache) tous les CSV. → {(saison, div): df}"""

    import requests

    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = {}
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) backtest-valuebet/1.0"
    )

    for season in SEASONS_ALL:
        for div in DIVISIONS:
            path = os.path.join(CACHE_DIR, f"{season}_{div}.csv")
            if not os.path.exists(path) or os.path.getsize(path) < 1000:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
                print(f"  téléchargement {url}")
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
            df = _read_csv(path)
            if df is not None and not df.empty:
                frames[(season, div)] = df

    return frames


def _read_csv(path: str):
    """Lecture robuste d'un CSV football-data (encodage/lignes cassées)."""

    for kwargs in (
        {"encoding": "latin-1"},
        {"encoding": "latin-1", "engine": "python", "on_bad_lines": "skip"},
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            break
        except Exception:
            df = None
    if df is None or "HomeTeam" not in df.columns:
        return None

    # Garantir toutes les colonnes utiles (NaN si absentes du fichier)
    for col in USE_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[USE_COLS].copy()

    # Lignes sans équipes ou sans score final = inutilisables
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    return df


# ══════════════════════════════════════════════════════════════
#  2. RECONSTRUCTION DES STATS D'ÉQUIPE (matchs précédents seuls)
# ══════════════════════════════════════════════════════════════

def _weighted_avg(rows, ref_date, take_last=WINDOW):
    """Moyennes pondérées (buts pour, buts contre) avec décote exp(-xi·jours).

    rows : liste chronologique de (date, gf, ga). On prend les
    `take_last` plus récents, pondérés par leur ancienneté au
    moment du match de référence — comme PoissonConfig.TIME_DECAY_XI.
    """

    recent = rows[-take_last:]
    sw = swf = swa = 0.0
    for d, gf, ga in recent:
        w = math.exp(-XI * max((ref_date - d).days, 0))
        sw += w
        swf += w * gf
        swa += w * ga
    if sw <= 0:
        return None
    return swf / sw, swa / sw


def build_team_stats(name, history, ref_date) -> TeamStats:
    """TeamStats "api" depuis l'historique strictement antérieur au match.

    history : {"all": [(date, gf, ga, pts)], "home": [(date, gf, ga)],
               "away": [(date, gf, ga)]} — listes chronologiques.
    """

    overall = _weighted_avg([(d, gf, ga) for d, gf, ga, _ in history["all"]], ref_date)
    home = _weighted_avg(history["home"], ref_date) if history["home"] else None
    away = _weighted_avg(history["away"], ref_date) if history["away"] else None

    avg_for, avg_against = overall

    # Splits manquants (équipe n'ayant pas encore joué à domicile/
    # extérieur) : dérivés de la moyenne globale avec les facteurs
    # standards de l'app (data_collector._estimate_stats_from_odds).
    if home is None:
        home = (avg_for * 1.12, avg_against * 0.90)
    if away is None:
        away = (avg_for * 0.88, avg_against * 1.10)

    last5 = history["all"][-5:]
    form = sum(p for _, _, _, p in last5) / len(last5) if last5 else 1.5

    stats = TeamStats(team_name=name, data_source="api")
    stats.avg_goals_scored = avg_for
    stats.avg_goals_conceded = avg_against
    stats.avg_goals_scored_home = home[0]
    stats.avg_goals_conceded_home = home[1]
    stats.avg_goals_scored_away = away[0]
    stats.avg_goals_conceded_away = away[1]
    stats.recent_form_score = form
    stats.points_per_game = form
    stats.matches_played = len(history["all"])
    return stats


def league_averages(frames) -> dict:
    """Moyenne de buts/match par (saison, div) → sert de moyenne
    "saison précédente" pour la saison suivante."""

    return {
        (season, div): float((df["FTHG"] + df["FTAG"]).mean())
        for (season, div), df in frames.items()
    }


def prepare_matches(frames):
    """Parcours chronologique par ligue : contexte + λ précalculés.

    Retourne la liste des matchs évaluables (≥ MIN_HISTORY matchs
    d'historique pour les deux équipes + cotes B365 valides).
    """

    predictor = PoissonPredictor()
    lg_avgs = league_averages(frames)
    prev_season = dict(zip(SEASONS_ALL[1:], SEASONS_ALL[:-1]))

    market_cache = {}  # (o1, ox, o2, league_avg arrondi) → λ marché

    matches = []
    n_seen = n_skip_hist = n_skip_odds = 0

    for div, league_key in DIVISIONS.items():
        league_info = SUPPORTED_LEAGUES[league_key]
        history = {}  # équipe → {"all": [...], "home": [...], "away": [...]}

        for season in SEASONS_BACKTEST:
            df = frames.get((season, div))
            if df is None:
                continue
            df = df.sort_values("Date", kind="stable")

            # Moyenne de la ligue = saison précédente (fallback config)
            lg_avg = lg_avgs.get(
                (prev_season[season], div), league_info["avg_goals"]
            )

            for row in df.itertuples(index=False):
                n_seen += 1
                date = row.Date
                ht, at = str(row.HomeTeam), str(row.AwayTeam)
                fthg, ftag = int(row.FTHG), int(row.FTAG)

                h_hist = history.setdefault(
                    ht, {"all": [], "home": [], "away": []})
                a_hist = history.setdefault(
                    at, {"all": [], "home": [], "away": []})

                o1, ox, o2 = _f(row.B365H), _f(row.B365D), _f(row.B365A)
                usable_odds = o1 > 1 and ox > 1 and o2 > 1
                enough_hist = (len(h_hist["all"]) >= MIN_HISTORY
                               and len(a_hist["all"]) >= MIN_HISTORY)

                if usable_odds and enough_hist:
                    ctx = MatchContext(
                        home_team=ht, away_team=at, league=league_key,
                        home_stats=build_team_stats(ht, h_hist, date),
                        away_stats=build_team_stats(at, a_hist, date),
                        odds={"1": o1, "X": ox, "2": o2},
                        league_avg_goals=lg_avg,
                        first_half_share=league_info["first_half_share"],
                        data_completeness=70.0,
                    )
                    stats_lams = predictor._lambdas_from_stats(ctx)
                    key = (o1, ox, o2, round(lg_avg, 3))
                    if key not in market_cache:
                        market_cache[key] = predictor._lambdas_from_market(ctx)
                    market_lams = market_cache[key]

                    if market_lams:
                        outcome = 0 if fthg > ftag else (1 if fthg == ftag else 2)
                        matches.append({
                            "div": div, "league": league_key,
                            "season": season, "date": date,
                            "home": ht, "away": at,
                            "fthg": fthg, "ftag": ftag,
                            "outcome": outcome,
                            "over25": 1 if fthg + ftag >= 3 else 0,
                            "b365": (o1, ox, o2),
                            "psc": (_f(row.PSCH), _f(row.PSCD), _f(row.PSCA)),
                            "stats_lams": stats_lams,
                            "market_lams": market_lams,
                        })
                    else:
                        n_skip_odds += 1
                elif not enough_hist:
                    n_skip_hist += 1
                else:
                    n_skip_odds += 1

                # Mise à jour de l'historique APRÈS l'évaluation
                h_pts = 3 if fthg > ftag else (1 if fthg == ftag else 0)
                a_pts = 3 if ftag > fthg else (1 if fthg == ftag else 0)
                h_hist["all"].append((date, fthg, ftag, h_pts))
                h_hist["home"].append((date, fthg, ftag))
                a_hist["all"].append((date, ftag, fthg, a_pts))
                a_hist["away"].append((date, ftag, fthg))

    print(f"  {n_seen} matchs lus — {len(matches)} évaluables "
          f"({n_skip_hist} skip historique, {n_skip_odds} skip cotes)")
    return matches


def _f(x) -> float:
    """Float robuste (NaN/None → 0)."""
    try:
        v = float(x)
        return v if v == v else 0.0  # NaN != NaN
    except (TypeError, ValueError):
        return 0.0


# ══════════════════════════════════════════════════════════════
#  3. PROBABILITÉS DU MODÈLE (blend manuel = _estimate_lambdas)
# ══════════════════════════════════════════════════════════════

def model_probs(match, weight):
    """(p1, px, p2, p_over25) — blend w×marché + (1-w)×stats, bornes,
    puis matrice Dixon-Coles du vrai modèle (_score_matrix)."""

    st, mk = match["stats_lams"], match["market_lams"]
    lam_h = weight * mk[0] + (1 - weight) * st[0]
    lam_a = weight * mk[1] + (1 - weight) * st[1]
    lam_h = max(PoissonConfig.MIN_LAMBDA, min(PoissonConfig.MAX_LAMBDA, lam_h))
    lam_a = max(PoissonConfig.MIN_LAMBDA, min(PoissonConfig.MAX_LAMBDA, lam_a))

    matrix = PoissonPredictor._score_matrix(lam_h, lam_a)
    p1 = px = p2 = over25 = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                p1 += p
            elif i == j:
                px += p
            else:
                p2 += p
            if i + j >= 3:
                over25 += p
    norm = p1 + px + p2
    return p1 / norm, px / norm, p2 / norm, over25


def log_loss_1x2(prob_rows, outcomes):
    total = 0.0
    for probs, y in zip(prob_rows, outcomes):
        total -= math.log(max(probs[y], 1e-12))
    return total / len(outcomes)


def brier_1x2(prob_rows, outcomes):
    total = 0.0
    for probs, y in zip(prob_rows, outcomes):
        total += sum((probs[k] - (1.0 if k == y else 0.0)) ** 2
                     for k in range(3))
    return total / len(outcomes)


# ══════════════════════════════════════════════════════════════
#  4. ÉVALUATIONS
# ══════════════════════════════════════════════════════════════

def grid_search_weight(train):
    """Courbe log-loss 1X2 sur le train pour chaque MARKET_WEIGHT."""

    outcomes = [m["outcome"] for m in train]
    curve = {}
    for w in MARKET_WEIGHT_GRID:
        rows = [model_probs(m, w)[:3] for m in train]
        curve[f"{w:.1f}"] = round(log_loss_1x2(rows, outcomes), 5)
        print(f"  MARKET_WEIGHT={w:.1f} → log-loss {curve[f'{w:.1f}']:.5f}")
    best = min(curve, key=curve.get)
    return curve, float(best)


def market_prob_rows(matches, key):
    """Probabilités no-vig Shin du marché pour chaque match (ou None)."""

    rows = []
    for m in matches:
        o = m[key]
        rows.append(novig_probs(list(o)) if all(v > 1 for v in o) else None)
    return rows


def evaluate_test(test, weight):
    """Brier + log-loss du modèle vs marché B365 et clôture Pinnacle,
    sur le sous-ensemble commun (les 3 jeux de probabilités présents)."""

    model_rows = [model_probs(m, weight) for m in test]
    b365_rows = market_prob_rows(test, "b365")
    psc_rows = market_prob_rows(test, "psc")

    idx = [i for i in range(len(test))
           if b365_rows[i] is not None and psc_rows[i] is not None]
    outcomes = [test[i]["outcome"] for i in idx]
    mod = [model_rows[i][:3] for i in idx]
    b365 = [b365_rows[i] for i in idx]
    psc = [psc_rows[i] for i in idx]

    metrics = {
        "n_matchs_compares": len(idx),
        "brier": {
            "modele": round(brier_1x2(mod, outcomes), 5),
            "marche_b365": round(brier_1x2(b365, outcomes), 5),
            "marche_cloture": round(brier_1x2(psc, outcomes), 5),
        },
        "logloss": {
            "modele": round(log_loss_1x2(mod, outcomes), 5),
            "marche_b365": round(log_loss_1x2(b365, outcomes), 5),
            "marche_cloture": round(log_loss_1x2(psc, outcomes), 5),
        },
    }
    return metrics, model_rows


def calibration_table(pred_probs, observed):
    """Table par déciles : prédite moyenne vs fréquence observée."""

    table = []
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        sel = [(p, o) for p, o in zip(pred_probs, observed)
               if lo <= p < hi or (k == 9 and p == 1.0)]
        if not sel:
            table.append({"decile": f"{lo:.1f}-{hi:.1f}", "n": 0,
                          "p_pred_moyenne": None, "freq_observee": None})
            continue
        table.append({
            "decile": f"{lo:.1f}-{hi:.1f}",
            "n": len(sel),
            "p_pred_moyenne": round(sum(p for p, _ in sel) / len(sel), 4),
            "freq_observee": round(sum(o for _, o in sel) / len(sel), 4),
        })
    return table


def value_threshold(odds):
    """Seuil de value progressif de l'app (base 5% × multiplicateur)."""

    for max_odds, mult in ValueBetConfig.VALUE_THRESHOLD_MULTIPLIERS:
        if odds < max_odds:
            return ValueBetConfig.MIN_VALUE_THRESHOLD * mult
    return ValueBetConfig.MIN_VALUE_THRESHOLD * \
        ValueBetConfig.VALUE_THRESHOLD_MULTIPLIERS[-1][1]


def simulate_strategy(test, model_rows):
    """Stratégie value sur le test : value = p_modèle × cote_B365 − 1,
    seuils progressifs, cotes [1.30, 6.00], quart-Kelly plafonné 2%
    d'un bankroll fixe. Au plus un pari (meilleure value) par match.
    CLV vs cote juste Pinnacle clôture (no-vig Shin)."""

    bets = []
    for m, probs in zip(test, model_rows):
        candidates = []
        for k in range(3):
            o = m["b365"][k]
            p = probs[k]
            if not (ValueBetConfig.MIN_ODDS <= o <= ValueBetConfig.MAX_ODDS):
                continue
            value = p * o - 1
            if value < value_threshold(o):
                continue
            if value > ValueBetConfig.MAX_PLAUSIBLE_VALUE:
                continue  # garde-fou de l'app : value implausible
            candidates.append((value, k, o, p))
        if not candidates:
            continue

        value, k, o, p = max(candidates)
        kelly = (p * o - 1) / (o - 1)
        stake = min(KELLY_FRACTION * kelly * BANKROLL, MAX_STAKE)
        won = (m["outcome"] == k)
        profit = stake * (o - 1) if won else -stake

        clv = None
        psc = m["psc"]
        if all(v > 1 for v in psc):
            fair = novig_probs(list(psc))
            if fair and fair[k] > 0:
                clv = o * fair[k] - 1  # cote_prise / cote_juste_clôture − 1

        bets.append({"odds": o, "value": value, "stake": stake,
                     "won": won, "profit": profit, "clv": clv,
                     "season": m["season"], "league": m["league"]})

    if not bets:
        return {"n_paris": 0}

    staked = sum(b["stake"] for b in bets)
    profit = sum(b["profit"] for b in bets)
    clvs = [b["clv"] for b in bets if b["clv"] is not None]

    def bucket_stats(sel):
        st = sum(b["stake"] for b in sel)
        pf = sum(b["profit"] for b in sel)
        return {
            "n_paris": len(sel),
            "roi_pct": round(100 * pf / st, 2) if st else None,
            "profit": round(pf, 0),
            "win_rate": round(sum(b["won"] for b in sel) / len(sel), 4),
        }

    par_seuil = {}
    bands = [(ValueBetConfig.MIN_ODDS, 2.50, "cotes_1.30-2.50_seuil_5pct"),
             (2.50, 4.00, "cotes_2.50-4.00_seuil_8pct"),
             (4.00, 6.00, "cotes_4.00-6.00_seuil_12pct")]
    for lo, hi, label in bands:
        sel = [b for b in bets if lo <= b["odds"] < hi or
               (hi == 6.00 and b["odds"] == 6.00)]
        if sel:
            par_seuil[label] = bucket_stats(sel)

    return {
        "n_paris": len(bets),
        "mise_totale": round(staked, 0),
        "roi_pct": round(100 * profit / staked, 2),
        "profit": round(profit, 0),
        "win_rate": round(sum(b["won"] for b in bets) / len(bets), 4),
        "cote_moyenne": round(sum(b["odds"] for b in bets) / len(bets), 2),
        "value_moyenne_pct": round(
            100 * sum(b["value"] for b in bets) / len(bets), 2),
        "clv_moyen_pct": round(100 * sum(clvs) / len(clvs), 2) if clvs else None,
        "clv_positif_pct": round(
            100 * sum(1 for c in clvs if c > 0) / len(clvs), 1) if clvs else None,
        "n_paris_avec_clv": len(clvs),
        "par_seuil": par_seuil,
    }


def empirical_league_stats(frames):
    """Part réelle des buts en 1ère MT et tirs cadrés par but, par ligue
    (saisons du backtest uniquement)."""

    half_share, sot_per_goal = {}, {}
    for div, league_key in DIVISIONS.items():
        parts = [frames[(s, div)] for s in SEASONS_BACKTEST
                 if (s, div) in frames]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)

        ht = df.dropna(subset=["HTHG", "HTAG"])
        ft_goals = float((ht["FTHG"] + ht["FTAG"]).sum())
        h1_goals = float((ht["HTHG"] + ht["HTAG"]).sum())
        half_share[league_key] = {
            "part_1ere_mt": round(h1_goals / ft_goals, 4) if ft_goals else None,
            "config_app": SUPPORTED_LEAGUES[league_key]["first_half_share"],
            "n_matchs": int(len(ht)),
        }

        st = df.dropna(subset=["HST", "AST"])
        goals = float((st["FTHG"] + st["FTAG"]).sum())
        sots = float((st["HST"] + st["AST"]).sum())
        sot_per_goal[league_key] = {
            "sot_par_but": round(sots / goals, 3) if goals else None,
            "config_app": PoissonConfig.SOT_PER_GOAL,
            "n_matchs": int(len(st)),
        }
    return half_share, sot_per_goal


# ══════════════════════════════════════════════════════════════
#  5. PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    original_weight = PoissonConfig.MARKET_WEIGHT  # jamais muté, mais on vérifie

    print("1) Téléchargement / cache des CSV football-data.co.uk ...")
    frames = download_all()
    print(f"   {len(frames)} fichiers chargés")

    print("2) Reconstruction chronologique des stats + λ précalculés ...")
    matches = prepare_matches(frames)

    train = [m for m in matches if m["season"] in SEASONS_TRAIN]
    test = [m for m in matches if m["season"] in SEASONS_TEST]
    print(f"   train {len(train)} matchs (2122-2324) — "
          f"test {len(test)} matchs (2425-2526)")

    print("3) Grille MARKET_WEIGHT sur le train (log-loss 1X2) ...")
    curve, best_w = grid_search_weight(train)
    print(f"   meilleur poids marché : {best_w:.1f}")

    print("4) Test : modèle vs marché no-vig (Shin) ...")
    metrics, model_rows = evaluate_test(test, best_w)

    print("5) Calibration par déciles ...")
    calib_home = calibration_table(
        [r[0] for r in model_rows], [m["outcome"] == 0 for m in test])
    calib_over = calibration_table(
        [r[3] for r in model_rows], [m["over25"] for m in test])

    print("6) Stratégie value quart-Kelly + CLV ...")
    strategy = simulate_strategy(test, model_rows)

    print("7) Stats empiriques par ligue (mi-temps, tirs cadrés) ...")
    half_share, sot_per_goal = empirical_league_stats(frames)

    assert PoissonConfig.MARKET_WEIGHT == original_weight, \
        "PoissonConfig.MARKET_WEIGHT a été altéré"

    results = {
        "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "football-data.co.uk (B365 ouverture, Pinnacle clôture)",
        "saisons_train": sorted(SEASONS_TRAIN),
        "saisons_test": sorted(SEASONS_TEST),
        "ligues": list(DIVISIONS.values()),
        "n_matchs_train": len(train),
        "n_matchs_test": len(test),
        "poids_marche": {
            "courbe_logloss_train": curve,
            "meilleur": best_w,
            "valeur_app": original_weight,
        },
        "brier": metrics["brier"],
        "logloss": metrics["logloss"],
        "n_matchs_compares": metrics["n_matchs_compares"],
        "calibration": {"home_win": calib_home, "over25": calib_over},
        "strategie": strategy,
        "mi_temps_par_ligue": half_share,
        "sot_par_but_par_ligue": sot_per_goal,
        "duree_backtest_s": round(time.time() - t0, 1),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    _print_summary(results)
    print(f"\nRésultats sauvés → {RESULTS_PATH}")
    print(f"Durée totale : {results['duree_backtest_s']}s")


def _print_summary(r):
    print("\n" + "═" * 60)
    print(" RÉSUMÉ DU BACKTEST")
    print("═" * 60)
    print(f"Matchs : {r['n_matchs_train']} train / {r['n_matchs_test']} test")

    print("\nGrille MARKET_WEIGHT (log-loss train) :")
    for w, ll in r["poids_marche"]["courbe_logloss_train"].items():
        star = "  ← meilleur" if float(w) == r["poids_marche"]["meilleur"] else ""
        print(f"  w={w} : {ll:.5f}{star}")

    b, l = r["brier"], r["logloss"]
    print(f"\nTest ({r['n_matchs_compares']} matchs) — Brier / log-loss 1X2 :")
    print(f"  Modèle           : {b['modele']:.5f} / {l['modele']:.5f}")
    print(f"  Marché B365      : {b['marche_b365']:.5f} / {l['marche_b365']:.5f}")
    print(f"  Marché clôture   : {b['marche_cloture']:.5f} / {l['marche_cloture']:.5f}")

    s = r["strategie"]
    if s.get("n_paris"):
        print(f"\nStratégie value (test) : {s['n_paris']} paris, "
              f"ROI {s['roi_pct']}%, profit {s['profit']:.0f} FCFA")
        print(f"  win rate {100 * s['win_rate']:.1f}%, cote moy {s['cote_moyenne']}, "
              f"value moy {s['value_moyenne_pct']}%")
        print(f"  CLV moyen {s['clv_moyen_pct']}% — CLV>0 sur "
              f"{s['clv_positif_pct']}% des paris ({s['n_paris_avec_clv']} avec clôture)")
        for label, st in s.get("par_seuil", {}).items():
            print(f"    {label} : {st['n_paris']} paris, ROI {st['roi_pct']}%")
    else:
        print("\nStratégie value : aucun pari déclenché")

    print("\nPart des buts en 1ère mi-temps (réel vs config) :")
    for lg, v in r["mi_temps_par_ligue"].items():
        print(f"  {lg:16s} : {v['part_1ere_mt']:.3f} (config {v['config_app']})")

    print("\nTirs cadrés par but (réel vs config 3.1) :")
    for lg, v in r["sot_par_but_par_ligue"].items():
        print(f"  {lg:16s} : {v['sot_par_but']:.2f}")

    print("\nCalibration P(victoire domicile) — déciles non vides :")
    for row in r["calibration"]["home_win"]:
        if row["n"]:
            print(f"  {row['decile']} : prédite {row['p_pred_moyenne']:.3f} "
                  f"vs observée {row['freq_observee']:.3f} (n={row['n']})")


if __name__ == "__main__":
    main()
