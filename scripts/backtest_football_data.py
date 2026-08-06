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
from modules.elo_rating import EloRatingSystem                                    # noqa: E402
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

# La grille s'arrêtait à 0.9 — et 0.9 ressortait comme « optimal »
# alors que c'était simplement la BORNE. En l'étendant, le log-loss
# continue de baisser jusqu'à 1.0 (marché pur), sur train ET test :
# la composante statistique maison DÉGRADE la prédiction 1X2. Ne
# jamais laisser un optimum se poser sur le bord d'une grille.
MARKET_WEIGHT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]

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
    # Over/Under de CLÔTURE (Pinnacle) : indispensable pour juger le
    # marché des totaux comme on juge le 1X2. Sans ligne de clôture,
    # pas de CLV — donc aucun moyen de savoir si une sélection était
    # bonne avant même de connaître le résultat.
    "PC>2.5", "PC<2.5",
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

    # « B365>2.5 » n'est pas un identifiant Python valide : itertuples
    # le renommerait en position (_11), ce qui casse silencieusement
    # dès que l'ordre des colonnes change. On renomme explicitement.
    df = df.rename(columns={"B365>2.5": "B365_OV25",
                            "B365<2.5": "B365_UN25",
                            "PC>2.5": "PC_OV25",
                            "PC<2.5": "PC_UN25"})
    return df


# ══════════════════════════════════════════════════════════════
#  1bis. HISTORIQUE xG MATCH PAR MATCH (Understat via soccerdata)
# ══════════════════════════════════════════════════════════════
#
# Le backtest tournait sur les BUTS seuls. Or l'app mélange xG et buts
# (XG_BLEND) dès que le xG est disponible : la courbe de poids marché
# mesurée sans xG ne décrivait donc pas le modèle réellement servi.
#
# APPARIEMENT DES NOMS — le point dangereux. football-data écrit
# « Man City », Understat « Manchester City ». Plutôt que de se fier à
# une ressemblance de chaînes (la mémoire du projet garde la trace
# d'un « Paris Saint-Germain » apparié à « Paris FC »), on DÉDUIT la
# correspondance des données elles-mêmes : deux matchs qui partagent
# la date ET le score exact sont le même match. Les paires ainsi
# identifiées votent, le vote majoritaire donne le mapping, puis on
# revérifie TOUTES les rencontres. Un mapping faux se trahit
# immédiatement par des scores qui divergent.

UNDERSTAT_LEAGUES = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "I1": "ITA-Serie A",
    "D1": "GER-Bundesliga",
    "F1": "FRA-Ligue 1",
}

XG_CACHE_PATH = os.path.join(CACHE_DIR, "xg_par_match.json")

# Sous ce taux de concordance des scores, on refuse le xG de la
# ligue-saison entière : mieux vaut aucun xG qu'un xG attribué à la
# mauvaise équipe.
SEUIL_CONCORDANCE = 0.90


def _cle_match(date, home, away) -> str:
    return f"{str(date)[:10]}|{home}|{away}"


def charger_xg(frames) -> dict:
    """
    {(saison, div): {clé_match: (xg_domicile, xg_extérieur)}}

    Utilise le cache disque si présent (le scraping Understat prend
    plusieurs minutes).
    """

    if os.path.exists(XG_CACHE_PATH):
        try:
            with open(XG_CACHE_PATH, encoding="utf-8") as f:
                brut = json.load(f)
            out = {}
            for cle, d in brut.items():
                saison, div = cle.split("|")
                out[(saison, div)] = {k: tuple(v) for k, v in d.items()}
            total = sum(len(v) for v in out.values())
            print(f"  cache xG : {total} matchs sur "
                  f"{len(out)} ligues-saisons")
            return out
        except Exception as e:
            print(f"  cache xG illisible ({e}) — reconstruction")

    import warnings
    warnings.filterwarnings("ignore")
    import soccerdata as sd

    resultat = {}
    for div, sd_league in UNDERSTAT_LEAGUES.items():
        for saison in SEASONS_BACKTEST:
            fd = frames.get((saison, div))
            if fd is None:
                continue
            try:
                us = sd.Understat(leagues=sd_league, seasons=saison)
                udf = us.read_team_match_stats().reset_index()
            except Exception as e:
                print(f"  {div} {saison} : xG indisponible "
                      f"({type(e).__name__})")
                continue

            mapping, taux, n = _deduire_mapping(fd, udf)
            if taux < SEUIL_CONCORDANCE:
                print(f"  {div} {saison} : REFUSÉ — concordance des "
                      f"scores {taux:.0%} sur {n} matchs")
                continue

            table = {}
            for u in udf.itertuples(index=False):
                h = mapping.get(u.home_team)
                a = mapping.get(u.away_team)
                if not h or not a:
                    continue
                try:
                    xh, xa = float(u.home_xg), float(u.away_xg)
                except (TypeError, ValueError):
                    continue
                if xh != xh or xa != xa:      # NaN
                    continue
                table[_cle_match(u.date, h, a)] = (xh, xa)

            resultat[(saison, div)] = table
            print(f"  {div} {saison} : {len(table)} matchs avec xG "
                  f"(concordance {taux:.0%})")

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(XG_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({f"{s}|{d}": v for (s, d), v in resultat.items()},
                      f)
    except OSError:
        pass

    return resultat


def _deduire_mapping(fd, udf):
    """
    Correspondance nom Understat → nom football-data, déduite des
    (date, score) uniques, puis validée sur toutes les rencontres.

    Retourne (mapping, taux_de_concordance, n_verifies).
    """

    from collections import defaultdict, Counter

    # Index football-data : (date, score) → [(home, away)]
    par_date_score = defaultdict(list)
    for r in fd.itertuples(index=False):
        cle = (str(r.Date)[:10], int(r.FTHG), int(r.FTAG))
        par_date_score[cle].append((str(r.HomeTeam), str(r.AwayTeam)))

    votes_h = defaultdict(Counter)
    votes_a = defaultdict(Counter)

    for u in udf.itertuples(index=False):
        try:
            hg, ag = int(u.home_goals), int(u.away_goals)
        except (TypeError, ValueError):
            continue
        # ±1 jour : Understat horodate en UTC, football-data en local
        for delta in (0, -1, 1):
            d = pd.Timestamp(u.date) + pd.Timedelta(days=delta)
            cands = par_date_score.get((str(d)[:10], hg, ag), [])
            if len(cands) == 1:            # appariement NON ambigu
                fh, fa = cands[0]
                votes_h[u.home_team][fh] += 1
                votes_a[u.away_team][fa] += 1
                break

    # Un nom Understat vote comme domicile ET comme extérieur :
    # on additionne les deux urnes.
    urnes = defaultdict(Counter)
    for nom, c in votes_h.items():
        urnes[nom].update(c)
    for nom, c in votes_a.items():
        urnes[nom].update(c)

    mapping = {nom: c.most_common(1)[0][0]
               for nom, c in urnes.items() if c}

    # VALIDATION : on rejoue toutes les rencontres avec ce mapping et
    # on exige que les scores concordent.
    scores_fd = {}
    for r in fd.itertuples(index=False):
        scores_fd[_cle_match(r.Date, str(r.HomeTeam), str(r.AwayTeam))] = \
            (int(r.FTHG), int(r.FTAG))

    ok = total = 0
    for u in udf.itertuples(index=False):
        h, a = mapping.get(u.home_team), mapping.get(u.away_team)
        if not h or not a:
            continue
        try:
            attendu = (int(u.home_goals), int(u.away_goals))
        except (TypeError, ValueError):
            continue
        for delta in (0, -1, 1):
            d = pd.Timestamp(u.date) + pd.Timedelta(days=delta)
            trouve = scores_fd.get(_cle_match(d, h, a))
            if trouve is not None:
                total += 1
                ok += int(trouve == attendu)
                break

    taux = (ok / total) if total else 0.0
    return mapping, taux, total


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


def build_team_stats(name, history, ref_date, avec_xg=False,
                     avec_sot=False) -> TeamStats:
    """TeamStats "api" depuis l'historique strictement antérieur au match.

    history : {"all": [(date, gf, ga, pts)], "home": [(date, gf, ga)],
               "away": [(date, gf, ga)],
               "xg_home": [(date, xgf, xga)], "xg_away": [...]}
              — listes chronologiques.

    avec_xg : remplit les champs xG de TeamStats comme le fait
    xg_provider en production, avec la MÊME fenêtre et la même décote
    temporelle que les buts. C'est indispensable : l'app mélange
    XG_BLEND × xG + (1−XG_BLEND) × buts, et comparer un xG calculé
    sur une fenêtre différente de celle des buts mesurerait l'écart
    de fenêtres, pas l'apport du xG.
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

    if avec_xg:
        xh = history.get("xg_home") or []
        xa = history.get("xg_away") or []
        # On exige un VRAI split des deux côtés : c'est la condition
        # que poisson_model teste (xg_home_split_real) pour décider
        # s'il réapplique l'avantage du terrain. Un xG « toutes
        # venues » servi comme un split ferait sauter cet avantage.
        if xh and xa:
            xg_h = _weighted_avg(xh, ref_date)
            xg_a = _weighted_avg(xa, ref_date)
            stats.xg_for_home, stats.xg_against_home = xg_h
            stats.xg_for_away, stats.xg_against_away = xg_a
            tous = sorted(xh + xa, key=lambda r: r[0])
            stats.xg_scored, stats.xg_conceded = _weighted_avg(
                tous, ref_date)
            stats.xg_available = True
            stats.xg_home_split_real = True
            stats.xg_away_split_real = True

    if avec_sot:
        # Tirs cadrés reconstruits comme les buts, depuis les matchs
        # PRÉCÉDENTS uniquement. Sans eux, _sot_lambdas renvoie None
        # et le modèle retombe sur l'approximation buts × SOT_PER_GOAL
        # — or l'app REFUSE de parier ce repli (sot_from_real_data).
        # Tester sans les reconstruire, ce serait tester un chemin
        # qu'elle n'emprunte jamais.
        sh = history.get("sot_home") or []
        sa = history.get("sot_away") or []
        tous = sorted(sh + sa, key=lambda r: r[0])
        if tous:
            moy = _weighted_avg(tous, ref_date)
            if moy:
                stats.avg_sot_for, stats.avg_sot_against = moy
                stats.sot_available = True
        # Splits par venue : c'est eux que le modèle corrigé utilise.
        if sh and sa:
            md = _weighted_avg(sh, ref_date)
            me = _weighted_avg(sa, ref_date)
            if md and me:
                stats.avg_sot_for_home, stats.avg_sot_against_home = md
                stats.avg_sot_for_away, stats.avg_sot_against_away = me
                stats.sot_venue_available = True

    return stats


def league_averages(frames) -> dict:
    """Moyenne de buts/match par (saison, div) → sert de moyenne
    "saison précédente" pour la saison suivante."""

    return {
        (season, div): float((df["FTHG"] + df["FTAG"]).mean())
        for (season, div), df in frames.items()
    }


def league_avg_sot(frames) -> dict:
    """Tirs cadrés moyens par équipe et par match, par (saison, div)."""

    out = {}
    for (season, div), df in frames.items():
        d = df.dropna(subset=["HST", "AST"])
        if len(d) < 20:
            continue
        out[(season, div)] = float((d["HST"] + d["AST"]).mean() / 2)
    return out


def prepare_matches(frames, xg_par_match=None, avec_xg=False,
                    avec_sot=False):
    """Parcours chronologique par ligue : contexte + λ précalculés.

    Retourne la liste des matchs évaluables (≥ MIN_HISTORY matchs
    d'historique pour les deux équipes + cotes B365 valides).

    avec_xg : injecte le xG des matchs PRÉCÉDENTS dans TeamStats.
    Comme pour les buts, le xG du match en cours n'est ajouté à
    l'historique qu'APRÈS l'évaluation — aucune anticipation.
    """

    xg_par_match = xg_par_match or {}
    predictor = PoissonPredictor()
    lg_avgs = league_averages(frames)
    lg_sot = league_avg_sot(frames)
    # Moyennes de tirs cadrés PAR VENUE, dénominateur du modèle corrigé
    lg_sot_dom, lg_sot_ext = {}, {}
    for (_s, _d), _df in frames.items():
        _x = _df.dropna(subset=["HST", "AST"])
        if len(_x) >= 20:
            lg_sot_dom[(_s, _d)] = float(_x["HST"].mean())
            lg_sot_ext[(_s, _d)] = float(_x["AST"].mean())
    prev_season = dict(zip(SEASONS_ALL[1:], SEASONS_ALL[:-1]))

    market_cache = {}  # (o1, ox, o2, league_avg arrondi) → λ marché

    matches = []
    n_seen = n_skip_hist = n_skip_odds = 0

    for div, league_key in DIVISIONS.items():
        league_info = SUPPORTED_LEAGUES[league_key]
        history = {}  # équipe → {"all": [...], "home": [...], "away": [...]}

        # Elo EN MARCHE AVANT, propre à la ligue. Deux précautions
        # méthodologiques indispensables :
        #  • clubelo neutralisé — ClubElo ne fournit que les ratings
        #    D'AUJOURD'HUI ; les injecter dans un backtest 2021-2026
        #    serait de la triche par anticipation.
        #  • ratings vidés — sinon les ratings du disque (appris sur
        #    des matchs postérieurs) contamineraient le passé.
        # L'Elo simulé est donc celui du chemin de REPLI de l'app
        # (estimé depuis les cotes puis mis à jour match après match),
        # pas son chemin ClubElo — limite assumée et documentée.
        elo = EloRatingSystem()
        elo.clubelo = None   # None, pas False : predict() teste `is not None`
        elo.ratings = {}

        for season in SEASONS_BACKTEST:
            df = frames.get((season, div))
            if df is None:
                continue
            df = df.sort_values("Date", kind="stable")
            xg_saison = xg_par_match.get((season, div), {})

            # Moyenne de la ligue = saison précédente (fallback config)
            lg_avg = lg_avgs.get(
                (prev_season[season], div), league_info["avg_goals"]
            )

            for row in df.itertuples(index=False):
                n_seen += 1
                date = row.Date
                ht, at = str(row.HomeTeam), str(row.AwayTeam)
                fthg, ftag = int(row.FTHG), int(row.FTAG)

                vide = {"all": [], "home": [], "away": [],
                        "xg_home": [], "xg_away": [],
                        "sot_home": [], "sot_away": []}
                h_hist = history.setdefault(
                    ht, {k: list(v) for k, v in vide.items()})
                a_hist = history.setdefault(
                    at, {k: list(v) for k, v in vide.items()})

                o1, ox, o2 = _f(row.B365H), _f(row.B365D), _f(row.B365A)
                usable_odds = o1 > 1 and ox > 1 and o2 > 1
                enough_hist = (len(h_hist["all"]) >= MIN_HISTORY
                               and len(a_hist["all"]) >= MIN_HISTORY)

                if usable_odds and enough_hist:
                    # Cotes Over/Under 2.5 : l'app les passe au
                    # contexte, ce qui affine nettement le λ total
                    # ajusté au marché. Les omettre faisait dévier
                    # λ_marché de 0.103 en moyenne.
                    odds_ctx = {"1": o1, "X": ox, "2": o2}
                    o_ov = _f(getattr(row, "B365_OV25", 0))
                    o_un = _f(getattr(row, "B365_UN25", 0))
                    if o_ov > 1 and o_un > 1:
                        odds_ctx["over_2_5"] = o_ov
                        odds_ctx["under_2_5"] = o_un

                    ctx = MatchContext(
                        home_team=ht, away_team=at, league=league_key,
                        home_stats=build_team_stats(
                            ht, h_hist, date, avec_xg=avec_xg,
                            avec_sot=avec_sot),
                        away_stats=build_team_stats(
                            at, a_hist, date, avec_xg=avec_xg,
                            avec_sot=avec_sot),
                        odds=odds_ctx,
                        league_avg_goals=lg_avg,
                        first_half_share=league_info["first_half_share"],
                        league_avg_goals_home=league_info.get(
                            "avg_goals_home", 0.0),
                        league_avg_goals_away=league_info.get(
                            "avg_goals_away", 0.0),
                        # Moyenne de tirs cadrés de la SAISON
                        # PRÉCÉDENTE — même précaution que pour les
                        # buts : utiliser celle de la saison en cours
                        # ferait entrer des matchs non encore joués.
                        league_avg_sot=lg_sot.get(
                            (prev_season[season], div), 0.0),
                        league_avg_sot_home=lg_sot_dom.get(
                            (prev_season[season], div), 0.0),
                        league_avg_sot_away=lg_sot_ext.get(
                            (prev_season[season], div), 0.0),
                        data_completeness=70.0,
                    )
                    stats_lams = predictor._lambdas_from_stats(ctx)
                    # La clé DOIT inclure les cotes Over/Under :
                    # _lambdas_from_market les lit aussi (elles sont
                    # posées sur odds_ctx juste au-dessus, et les
                    # omettre déviait λ_marché de 0.103).
                    #
                    # Sans elles, deux matchs au même 1X2 mais à
                    # totaux différents partageaient le même λ.
                    # Mesuré : 1322 collisions sur 8500 matchs
                    # (15,6 %), erreur moyenne 0,159 sur λ total.
                    # Le 1X2 restait presque intact — λ est ajusté
                    # dessus — mais P(over 2.5) dérivait, et c'est
                    # elle qui déclenche les paris Over/Under.
                    key = (o1, ox, o2, round(lg_avg, 3),
                           odds_ctx.get("over_2_5"),
                           odds_ctx.get("under_2_5"))
                    if key not in market_cache:
                        market_cache[key] = predictor._lambdas_from_market(ctx)
                    market_lams = market_cache[key]

                    if market_lams:
                        # Elo comme l'app : ratings estimés depuis les
                        # cotes du match, puis prédiction. L'appren-
                        # tissage du résultat vient APRÈS (plus bas),
                        # donc aucune information du futur n'entre ici.
                        elo.estimate_rating_from_odds(
                            ht, o1, o2, is_home=True)
                        elo.estimate_rating_from_odds(
                            at, o2, o1, is_home=False)
                        ep = elo.predict(ht, at)

                        outcome = 0 if fthg > ftag else (1 if fthg == ftag else 2)
                        matches.append({
                            "div": div, "league": league_key,
                            "season": season, "date": date,
                            "home": ht, "away": at,
                            "fthg": fthg, "ftag": ftag,
                            "outcome": outcome,
                            "over25": 1 if fthg + ftag >= 3 else 0,
                            # Marchés secondaires : issue réelle et
                            # cotes d'ouverture / de clôture.
                            "btts": 1 if (fthg > 0 and ftag > 0) else 0,
                            "ou_open": (o_ov, o_un),
                            "ou_close": (_f(getattr(row, "PC_OV25", 0)),
                                         _f(getattr(row, "PC_UN25", 0))),
                            # Issues réelles des marchés par équipe,
                            # par mi-temps et des tirs cadrés — pour
                            # mesurer la calibration du modèle là où
                            # aucune cote historique n'existe.
                            "buts": (fthg, ftag),
                            "first_half_share":
                                league_info["first_half_share"],
                            # λ tirs cadrés par la VRAIE méthode de
                            # l'app (attaque × défense / moyenne de
                            # ligue). None si les statistiques de tirs
                            # manquent — c'est alors le repli que
                            # l'app refuse de parier.
                            "sot_lams": (
                                PoissonPredictor._sot_lambdas(ctx)
                                if avec_sot else None),
                            "mi_temps": (_i(getattr(row, "HTHG", None)),
                                         _i(getattr(row, "HTAG", None))),
                            "sot": (_i(getattr(row, "HST", None)),
                                    _i(getattr(row, "AST", None))),
                            "b365": (o1, ox, o2),
                            "psc": (_f(row.PSCH), _f(row.PSCD), _f(row.PSCA)),
                            "stats_lams": stats_lams,
                            "market_lams": market_lams,
                            "elo_probs": (ep.prob_home_win, ep.prob_draw,
                                          ep.prob_away_win),
                            # Toujours « estimé » ici : ClubElo est
                            # neutralisé (ses ratings sont ceux
                            # d'aujourd'hui → anticipation).
                            "elo_source": getattr(ep, "elo_source", "estimé"),
                        })
                    else:
                        n_skip_odds += 1
                elif not enough_hist:
                    n_skip_hist += 1
                else:
                    n_skip_odds += 1

                # Apprentissage APRÈS l'évaluation : l'Elo n'a jamais
                # vu ce résultat au moment où il a prédit ce match.
                elo.record_result(ht, at, fthg, ftag)

                # Mise à jour de l'historique APRÈS l'évaluation
                h_pts = 3 if fthg > ftag else (1 if fthg == ftag else 0)
                a_pts = 3 if ftag > fthg else (1 if fthg == ftag else 0)
                h_hist["all"].append((date, fthg, ftag, h_pts))
                h_hist["home"].append((date, fthg, ftag))
                a_hist["all"].append((date, ftag, fthg, a_pts))
                a_hist["away"].append((date, ftag, fthg))

                # xG du match qui vient d'être évalué — ajouté APRÈS,
                # exactement comme les buts.
                xg = xg_saison.get(_cle_match(date, ht, at))
                if xg:
                    xg_h, xg_a = xg
                    h_hist["xg_home"].append((date, xg_h, xg_a))
                    a_hist["xg_away"].append((date, xg_a, xg_h))

                # Tirs cadrés — même règle : après l'évaluation.
                hst, ast = _i(getattr(row, "HST", None)), \
                    _i(getattr(row, "AST", None))
                if hst is not None and ast is not None:
                    h_hist["sot_home"].append((date, hst, ast))
                    a_hist["sot_away"].append((date, ast, hst))

    print(f"  {n_seen} matchs lus — {len(matches)} évaluables "
          f"({n_skip_hist} skip historique, {n_skip_odds} skip cotes)")
    return matches


def _i(x):
    """Entier robuste, ou None (NaN, vide, illisible)."""
    try:
        v = float(x)
        return int(v) if v == v else None
    except (TypeError, ValueError):
        return None


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

def model_probs(match, weight, avec_elo: bool = None):
    """
    (p1, px, p2, p_over25) exactement comme l'app.

    Deux étages, comme dans la vraie chaîne :
      1. λ = w×marché + (1-w)×stats → matrice Dixon-Coles (Poisson)
      2. BLEND ELO, mais UNIQUEMENT si l'Elo est indépendant
         (ClubElo), exactement comme value_detector.analyze_match.

    L'Elo simulé ici est celui du chemin de REPLI (estimé depuis les
    cotes), donc circulaire : l'app ne le mélange plus, et le
    backtest non plus — d'où avec_elo=False par défaut. Mesuré sur
    8500 matchs, le mélanger dégradait le log-loss de façon monotone
    (0.97156 sans Elo → 0.97683 à 40 %).

    avec_elo=True force le mélange, pour quantifier ce que coûterait
    (ou rapporterait) un Elo à cette pondération.
    """
    if avec_elo is None:
        avec_elo = match.get("elo_source") == "clubelo"

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
    p1, px, p2 = p1 / norm, px / norm, p2 / norm

    elo = match.get("elo_probs")
    if avec_elo and elo:
        wp, we = ValueBetConfig.POISSON_WEIGHT, ValueBetConfig.ELO_WEIGHT
        p1 = wp * p1 + we * elo[0]
        px = wp * px + we * elo[1]
        p2 = wp * p2 + we * elo[2]
        tot = p1 + px + p2
        if tot > 0:
            p1, px, p2 = p1 / tot, px / tot, p2 / tot

    return p1, px, p2, over25


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

def _test_apparie_xg(test_sans, test_avec, weight):
    """
    Le xG améliore-t-il vraiment, ou est-ce du bruit ?

    Les deux variantes portent sur LES MÊMES matchs, dans le même
    ordre : on compare donc match par match (test apparié), ce qui
    élimine la variance due au tirage des rencontres. Un écart moyen
    de 0,0009 de log-loss sur ~3400 matchs peut très bien n'être
    qu'une fluctuation — seul l'écart-type des différences le dit.
    """

    if len(test_sans) != len(test_avec):
        return None

    diffs = []
    for ms, ma in zip(test_sans, test_avec):
        if ms["date"] != ma["date"] or ms["home"] != ma["home"]:
            return None                      # alignement rompu
        o = ms["outcome"]
        ps = max(model_probs(ms, weight)[o], 1e-12)
        pa = max(model_probs(ma, weight)[o], 1e-12)
        # log-loss = −ln(p) ; positif = la variante xG fait mieux
        diffs.append(math.log(pa) - math.log(ps))

    n = len(diffs)
    if n < 2:
        return None
    moy = sum(diffs) / n
    var = sum((d - moy) ** 2 for d in diffs) / (n - 1)
    et = math.sqrt(var / n) if var > 0 else 0.0
    t = (moy / et) if et > 0 else 0.0
    return {
        "n_matchs": n,
        "ecart_moyen": round(moy, 7),
        "erreur_type": round(et, 7),
        "t": round(t, 3),
        "significatif": abs(t) > 1.96,
        "lecture": ("positif = le xG ameliore la prediction ; "
                    "|t| > 1.96 = significatif a 5 %"),
    }


def par_saison(test, poids):
    """
    Le verdict tient-il sur CHAQUE saison de test, ou repose-t-il sur
    une seule ?

    Un résultat agrégé sur plusieurs saisons peut masquer une saison
    excellente et une catastrophique. La saison la plus récente
    compte particulièrement : c'est celle qui ressemble le plus aux
    matchs à venir.
    """

    out = {}
    saisons = sorted({m["season"] for m in test})
    for s in saisons:
        sous = [m for m in test if m["season"] == s]
        if len(sous) < 100:
            continue
        outcomes = [m["outcome"] for m in sous]
        b365 = market_prob_rows(sous, "b365")
        psc = market_prob_rows(sous, "psc")

        # DEUX BASES DISTINCTES, et il faut les séparer.
        #
        # Comparer un log-loss de modèle calculé sur TOUS les matchs à
        # un log-loss de cote calculé sur les seuls matchs ayant
        # ouverture ET clôture produit des colonnes qui ne parlent pas
        # des mêmes matchs. Ce projet s'est déjà fait prendre.
        #
        # Et la clôture n'est PAS toujours disponible : football-data
        # n'a publié la clôture Pinnacle que sur ~la moitié de la
        # saison 2025-26 (210 matchs sur 380 en Premier League), alors
        # que l'ouverture y est complète. Restreindre toute la
        # comparaison à ce demi-échantillon la biaiserait — c'est ce
        # qui faisait apparaître la clôture MOINS bonne que
        # l'ouverture, un résultat impossible en soi.
        #
        # Base principale : les matchs cotés à l'OUVERTURE (complets).
        # La clôture est rapportée à part, avec son propre effectif.
        idx = [i for i in range(len(sous)) if b365[i] is not None]
        if len(idx) < 100:
            continue
        idx_clo = [i for i in idx if psc[i] is not None]
        psc_ok = [psc[i] for i in idx_clo]
        y_clo = [outcomes[i] for i in idx_clo]
        sous = [sous[i] for i in idx]
        outcomes = [outcomes[i] for i in idx]
        b365 = [b365[i] for i in idx]
        y = outcomes

        rows_mod = [model_probs(m, poids)[:3] for m in sous]
        rows_pur = [model_probs(m, 1.0)[:3] for m in sous]

        # Le modèle bat-il le marché pur, sur CETTE saison ?
        d = []
        for m in sous:
            o = m["outcome"]
            pm = max(model_probs(m, poids)[o], 1e-12)
            pp = max(model_probs(m, 1.0)[o], 1e-12)
            d.append(math.log(pm) - math.log(pp))
        n = len(d)
        moy = sum(d) / n
        var = sum((v - moy) ** 2 for v in d) / (n - 1)
        et = math.sqrt(var / n) if var > 0 else 0.0
        t = (moy / et) if et > 0 else 0.0

        out[s] = {
            "n": n,
            "logloss_modele": round(log_loss_1x2(rows_mod, outcomes), 5),
            "logloss_marche_pur": round(log_loss_1x2(rows_pur, outcomes), 5),
            "logloss_b365": round(log_loss_1x2(b365, y), 5),
            "logloss_cloture": (round(log_loss_1x2(psc_ok, y_clo), 5)
                                if len(idx_clo) >= 100 else None),
            "n_avec_cloture": len(idx_clo),
            "note": ("modèle, marché pur et ouverture portent sur les "
                     "mêmes matchs (n) ; la clôture sur n_avec_cloture, "
                     "football-data ne l'ayant pas publiée partout"),
            "vs_marche_pur_t": round(t, 3),
            "bat_le_marche": bool(t > 1.96),
            "pire_que_le_marche": bool(t < -1.96),
        }
    return out


def courbe_logloss(matches, etiquette=""):
    """Log-loss 1X2 pour chaque MARKET_WEIGHT de la grille."""

    outcomes = [m["outcome"] for m in matches]
    curve = {}
    for w in MARKET_WEIGHT_GRID:
        rows = [model_probs(m, w)[:3] for m in matches]
        curve[f"{w:.2f}"] = round(log_loss_1x2(rows, outcomes), 5)
    if etiquette:
        for w, ll in curve.items():
            print(f"    w={w} → {ll:.5f}")
    return curve


def grid_search_weight(train):
    """
    Courbe log-loss 1X2 sur le train pour chaque MARKET_WEIGHT.

    ATTENTION À LA LECTURE : 1.0 est le BORD DROIT de la grille. Un
    minimum qui s'y pose n'est pas un optimum observé — c'est une
    borne. Et cette courbe est calculée EN ÉCHANTILLON ; seule la
    courbe test (hors échantillon) autorise une conclusion.
    """

    curve = courbe_logloss(train)
    for w, ll in curve.items():
        print(f"  MARKET_WEIGHT={w} → log-loss {ll:.5f}")
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

    print("1bis) Historique xG match par match (Understat) ...")
    xg_par_match = charger_xg(frames)

    # ── Variante SANS xG : reproduit le backtest précédent, et sert
    #    de référence pour mesurer ce que le xG apporte (ou non).
    print("2) Reconstruction chronologique — SANS xG ...")
    matches_sans = prepare_matches(frames, xg_par_match, avec_xg=False)
    train_s = [m for m in matches_sans if m["season"] in SEASONS_TRAIN]
    test_s = [m for m in matches_sans if m["season"] in SEASONS_TEST]

    print("2bis) Reconstruction chronologique — AVEC xG ...")
    matches = prepare_matches(frames, xg_par_match, avec_xg=True)

    train = [m for m in matches if m["season"] in SEASONS_TRAIN]
    test = [m for m in matches if m["season"] in SEASONS_TEST]
    print(f"   train {len(train)} matchs (2122-2324) — "
          f"test {len(test)} matchs (2425-2526)")

    print("3) Grille MARKET_WEIGHT sur le train (log-loss 1X2) ...")
    curve, best_w = grid_search_weight(train)
    print(f"   meilleur poids marché (train, AVEC xG) : {best_w:.2f}")

    # ── LE chiffre qui décide : la courbe HORS ÉCHANTILLON.
    #    La courbe train est en échantillon et son minimum se pose
    #    sur le bord de grille ; elle ne prouve rien à elle seule.
    print("3bis) Courbes HORS ÉCHANTILLON (test) ...")
    print("   sans xG :")
    curve_test_sans = courbe_logloss(test_s, etiquette="test sans xG")
    print("   avec xG :")
    curve_test_avec = courbe_logloss(test, etiquette="test avec xG")
    curve_train_sans = courbe_logloss(train_s)
    best_test_sans = float(min(curve_test_sans, key=curve_test_sans.get))
    best_test_avec = float(min(curve_test_avec, key=curve_test_avec.get))
    print(f"   meilleur poids HORS ÉCHANTILLON — "
          f"sans xG : {best_test_sans:.2f} · avec xG : "
          f"{best_test_avec:.2f}")

    # Comparaison à poids IDENTIQUE (celui de l'app) : isole l'effet
    # du xG de l'effet d'un changement de poids.
    w_app = original_weight
    cle_app = f"{w_app:.2f}"
    apport = None
    if cle_app in curve_test_sans and cle_app in curve_test_avec:
        apport = round(curve_test_sans[cle_app] - curve_test_avec[cle_app], 6)
        print(f"   à w={cle_app} (valeur de l'app), le xG change le "
              f"log-loss test de {apport:+.6f} "
              f"({'mieux' if apport > 0 else 'moins bien'})")

    # Test apparié : l'écart de 0,0009 de log-loss est-il un signal
    # ou du bruit ? Sans ce test, on lirait une amélioration là où il
    # n'y a qu'une fluctuation. Comparaison match par match, à poids
    # IDENTIQUE — sinon on mesure le changement de poids, pas le xG.
    apport_test = _test_apparie_xg(test_s, test, w_app)
    if apport_test:
        print(f"   test apparié à w={w_app} : écart moyen "
              f"{apport_test['ecart_moyen']:+.6f} par match, "
              f"t={apport_test['t']:+.2f}, "
              f"{'SIGNIFICATIF' if apport_test['significatif'] else 'NON significatif'}")

    print("4) Test : modèle vs marché no-vig (Shin) ...")
    metrics, model_rows = evaluate_test(test, best_w)
    # Comparaison sans/avec xG au poids de l'APP (0.90). L'évaluer au
    # poids retenu (1.00) donnerait deux fois le même chiffre : à
    # poids marché pur, la composante statistique — donc le xG — ne
    # pèse rien. Le piège est facile à ne pas voir.
    metrics_sans_app, _ = evaluate_test(test_s, w_app)
    metrics_avec_app, _ = evaluate_test(test, w_app)

    print("5) Calibration par déciles ...")
    calib_home = calibration_table(
        [r[0] for r in model_rows], [m["outcome"] == 0 for m in test])
    calib_over = calibration_table(
        [r[3] for r in model_rows], [m["over25"] for m in test])

    print("6) Stratégie value quart-Kelly + CLV ...")
    strategy = simulate_strategy(test, model_rows)

    print("6bis) Ventilation par saison de test ...")
    saisons_detail = par_saison(test, w_app)
    for s_, d_ in saisons_detail.items():
        v_ = ("BAT le marché" if d_["bat_le_marche"]
              else "PIRE" if d_["pire_que_le_marche"] else "indiscernable")
        print(f"   {s_} : {d_['n']} matchs · modèle "
              f"{d_['logloss_modele']:.5f} vs marché pur "
              f"{d_['logloss_marche_pur']:.5f}  t={d_['vs_marche_pur_t']:+.2f} "
              f"→ {v_}")

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
            # ── Hors échantillon : le seul juge recevable. La courbe
            #    train est en échantillon et son minimum se pose sur
            #    le bord droit de la grille (1.0), ce qui n'est pas un
            #    optimum observé mais une borne.
            "courbe_logloss_test_sans_xg": curve_test_sans,
            "courbe_logloss_test_avec_xg": curve_test_avec,
            "courbe_logloss_train_sans_xg": curve_train_sans,
            "meilleur_test_sans_xg": best_test_sans,
            "meilleur_test_avec_xg": best_test_avec,
        },
        "apport_xg": {
            "description": ("Écart de log-loss HORS ÉCHANTILLON entre "
                            "le modèle sans xG et avec xG, à poids "
                            "marché identique. Positif = le xG "
                            "améliore."),
            "poids_compare": original_weight,
            "gain_logloss_test": apport,
            "brier_test_sans_xg": metrics_sans_app["brier"]["modele"],
            "brier_test_avec_xg": metrics_avec_app["brier"]["modele"],
            "logloss_test_sans_xg": metrics_sans_app["logloss"]["modele"],
            "logloss_test_avec_xg": metrics_avec_app["logloss"]["modele"],
            "n_matchs_test": metrics_avec_app["n_matchs_compares"],
            "test_apparie": apport_test,
            "avertissement": ("À w=1.00 la composante statistique ne "
                              "pèse rien : sans xG et avec xG y donnent "
                              "le MÊME chiffre, par construction. Toute "
                              "comparaison doit se faire à poids < 1."),
        },
        "brier": metrics["brier"],
        "logloss": metrics["logloss"],
        "n_matchs_compares": metrics["n_matchs_compares"],
        "par_saison": saisons_detail,
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

    pm = r["poids_marche"]
    print("\nGrille MARKET_WEIGHT — log-loss (plus bas = mieux)")
    print(f"  {'poids':>6s} {'train sans':>11s} {'train avec':>11s} "
          f"{'TEST sans':>11s} {'TEST avec':>11s}")
    for w in pm["courbe_logloss_train"]:
        ts = pm["courbe_logloss_test_sans_xg"].get(w)
        ta = pm["courbe_logloss_test_avec_xg"].get(w)
        trs = pm["courbe_logloss_train_sans_xg"].get(w)
        tra = pm["courbe_logloss_train"].get(w)
        marque = ""
        if ta is not None and ta == min(
                pm["courbe_logloss_test_avec_xg"].values()):
            marque = "  ← meilleur hors échantillon"
        print(f"  {w:>6s} {trs:11.5f} {tra:11.5f} {ts:11.5f} "
              f"{ta:11.5f}{marque}")
    print(f"  Meilleur poids HORS ÉCHANTILLON — "
          f"sans xG : {pm['meilleur_test_sans_xg']:.2f} · "
          f"avec xG : {pm['meilleur_test_avec_xg']:.2f}")

    ax = r["apport_xg"]
    print(f"\nApport du xG (hors échantillon, {ax['n_matchs_test']} matchs, "
          f"à poids {ax['poids_compare']}) :")
    print(f"  log-loss  sans xG {ax['logloss_test_sans_xg']:.5f}  →  "
          f"avec xG {ax['logloss_test_avec_xg']:.5f}")
    print(f"  Brier     sans xG {ax['brier_test_sans_xg']:.5f}  →  "
          f"avec xG {ax['brier_test_avec_xg']:.5f}")
    if ax["gain_logloss_test"] is not None:
        signe = "AMÉLIORE" if ax["gain_logloss_test"] > 0 else "DÉGRADE"
        print(f"  → le xG {signe} de {abs(ax['gain_logloss_test']):.6f} "
              f"de log-loss")
    ta = ax.get("test_apparie")
    if ta:
        verdict = ("SIGNIFICATIF" if ta["significatif"]
                   else "NON significatif — indiscernable du bruit")
        print(f"  Test apparié ({ta['n_matchs']} matchs) : "
              f"écart {ta['ecart_moyen']:+.6f} ± {ta['erreur_type']:.6f}, "
              f"t={ta['t']:+.2f} → {verdict}")

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
