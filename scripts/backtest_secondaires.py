"""
═══════════════════════════════════════════════════════════════
 BACKTEST — championnats SECONDAIRES, avec et sans xG
═══════════════════════════════════════════════════════════════

Même moteur que scripts/backtest_football_data.py, appliqué là où
la question reste ouverte : les championnats secondaires.

POURQUOI — le backtest des 5 grands a conclu que la composante
statistique du modèle, même améliorée par le xG, reste indiscernable
de zéro face au marché (optimum hors échantillon à w≈0.99). Mais les
marchés des grands championnats sont les plus efficients du football.
L'hypothèse qui restait à tester : dans un championnat moins traité
par les bookmakers, le modèle a-t-il une part à jouer ?

DEUX QUESTIONS, PAS UNE :
  1. Le poids marché optimal descend-il sous 1.0 ? (le modèle
     a-t-il une valeur propre là-bas ?)
  2. Le xG améliore-t-il, et de combien ?

SOURCE xG — Understat ne couvre que les 5 grands et FBref refuse les
requêtes (HTTP 403) : l'historique vient d'API-Football, collecté par
scripts/collecte_xg_api.py. Couverture mesurée à 100 % sur trois
saisons pour les quatre championnats retenus.

Sortie : data/backtest_secondaires.json
"""

import os
import sys
import json
import time
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bt_base", os.path.join(ROOT, "scripts",
                            "backtest_football_data.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

from config import PoissonConfig                          # noqa: E402


# ── Périmètre ──────────────────────────────────────────────────
# Retenus : les 4 championnats à couverture xG mesurée à 100 % sur
# trois saisons, tous présents dans le scan de l'app.
# Écartés faute de xG historique : Grèce, Écosse et 2. Bundesliga
# (aucune donnée avant 2025) ; Eredivisie (88 %, trous irréguliers).
DIVISIONS_SEC = {
    "E1": "championship",
    "P1": "primeira_liga",
    "B1": "belgium_pro",
    "T1": "super_lig",
}

SEASONS_ALL = ["2223", "2324", "2425", "2526"]
SEASONS_BACKTEST = ["2324", "2425", "2526"]
SEASONS_TRAIN = {"2324", "2425"}
SEASONS_TEST = {"2526"}

XG_API_CACHE = os.path.join(ROOT, "data", "football_data",
                            "xg_api_secondaires.json")
RESULTS_PATH = os.path.join(ROOT, "data", "backtest_secondaires.json")


def charger_xg_api(frames) -> dict:
    """
    Cache API-Football → format du backtest, avec le MÊME appariement
    par (date, score) que pour Understat.

    Les noms d'API-Football et de football-data diffèrent autant que
    ceux d'Understat (« Nott'm Forest » contre « Nottingham Forest »).
    On ne les apparie donc jamais par ressemblance : deux matchs
    partageant la date et le score exact sont le même match, les
    paires non ambiguës votent, puis on revérifie toutes les
    rencontres. Un mapping faux se trahit par des scores qui divergent.
    """

    try:
        with open(XG_API_CACHE, encoding="utf-8") as f:
            brut = json.load(f)
    except (OSError, ValueError) as e:
        print(f"  cache xG API illisible ({e}) — lancer d'abord "
              f"scripts/collecte_xg_api.py")
        return {}

    resultat = {}
    for cle_bloc, matchs in brut.items():
        saison, div = cle_bloc.split("|")
        fd = frames.get((saison, div))
        if fd is None or not matchs:
            continue

        lignes = [
            {"date": pd.Timestamp(m["date"]),
             "home_team": m["home"], "away_team": m["away"],
             "home_goals": m["gh"], "away_goals": m["ga"],
             "home_xg": (m["xg"] or [None, None])[0],
             "away_xg": (m["xg"] or [None, None])[1]}
            for m in matchs.values()
            if m.get("gh") is not None and m.get("ga") is not None
        ]
        if not lignes:
            continue
        udf = pd.DataFrame(lignes)

        mapping, taux, n = bt._deduire_mapping(fd, udf)
        if taux < bt.SEUIL_CONCORDANCE:
            print(f"  {div} {saison} : REFUSÉ — concordance des scores "
                  f"{taux:.0%} sur {n} matchs")
            continue

        table = {}
        for u in udf.itertuples(index=False):
            h, a = mapping.get(u.home_team), mapping.get(u.away_team)
            if not h or not a:
                continue
            # PIÈGE : pandas transforme les None en NaN, et
            # `NaN is None` vaut False. Un test « is None » laissait
            # donc passer les matchs SANS xG avec la valeur nan —
            # laquelle contamine ensuite toute moyenne pondérée de
            # l'équipe (nan absorbe tout). Il faut tester x != x.
            try:
                xh, xa = float(u.home_xg), float(u.away_xg)
            except (TypeError, ValueError):
                continue
            if xh != xh or xa != xa:          # NaN
                continue
            table[bt._cle_match(u.date, h, a)] = (xh, xa)
        resultat[(saison, div)] = table
        print(f"  {div} {saison} : {len(table)}/{len(udf)} matchs avec "
              f"xG (concordance des scores {taux:.0%})")

    return resultat


def main():
    t0 = time.time()
    poids_app = PoissonConfig.MARKET_WEIGHT

    # On force le périmètre du moteur commun sur les secondaires.
    bt.DIVISIONS = DIVISIONS_SEC
    bt.SEASONS_ALL = SEASONS_ALL
    bt.SEASONS_BACKTEST = SEASONS_BACKTEST
    bt.SEASONS_TRAIN = SEASONS_TRAIN
    bt.SEASONS_TEST = SEASONS_TEST

    print("1) CSV football-data des championnats secondaires ...")
    frames = bt.download_all()
    print(f"   {len(frames)} fichiers")

    print("2) Historique xG (API-Football) ...")
    xg = charger_xg_api(frames)
    n_xg = sum(len(v) for v in xg.values())
    print(f"   {n_xg} matchs avec xG sur {len(xg)} ligues-saisons")
    if n_xg < 500:
        print("   PAS ASSEZ DE xG — arrêt. Lancer collecte_xg_api.py")
        return

    print("3) Reconstruction walk-forward — sans xG ...")
    m_sans = bt.prepare_matches(frames, xg, avec_xg=False)
    print("4) Reconstruction walk-forward — avec xG ...")
    m_avec = bt.prepare_matches(frames, xg, avec_xg=True)

    tr_s = [m for m in m_sans if m["season"] in SEASONS_TRAIN]
    te_s = [m for m in m_sans if m["season"] in SEASONS_TEST]
    tr_a = [m for m in m_avec if m["season"] in SEASONS_TRAIN]
    te_a = [m for m in m_avec if m["season"] in SEASONS_TEST]
    print(f"   train {len(tr_a)} — test {len(te_a)}")

    print("5) Courbes de poids marché ...")
    c_tr_s = bt.courbe_logloss(tr_s)
    c_tr_a = bt.courbe_logloss(tr_a)
    c_te_s = bt.courbe_logloss(te_s)
    c_te_a = bt.courbe_logloss(te_a)
    best_te_s = float(min(c_te_s, key=c_te_s.get))
    best_te_a = float(min(c_te_a, key=c_te_a.get))

    print("6) Tests appariés ...")
    apport = bt._test_apparie_xg(te_s, te_a, poids_app)

    # Le poids marché optimal descend-il RÉELLEMENT sous 1.0 ?
    # Un minimum de courbe ne suffit pas : il faut que l'écart
    # survive à sa propre incertitude, match par match.
    poids_vs_marche = _compare_poids(te_a, poids_app)

    metrics_a, rows_a = bt.evaluate_test(te_a, poids_app)
    metrics_s, _ = bt.evaluate_test(te_s, poids_app)
    strategie = bt.simulate_strategy(te_a, rows_a)

    res = {
        "genere_le": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "perimetre": "championnats secondaires",
        "ligues": list(DIVISIONS_SEC.values()),
        "source_xg": "API-Football (/fixtures/statistics)",
        "saisons_train": sorted(SEASONS_TRAIN),
        "saisons_test": sorted(SEASONS_TEST),
        "n_matchs_train": len(tr_a),
        "n_matchs_test": len(te_a),
        "n_matchs_avec_xg": n_xg,
        "poids_marche": {
            "courbe_train_sans_xg": c_tr_s,
            "courbe_train_avec_xg": c_tr_a,
            "courbe_test_sans_xg": c_te_s,
            "courbe_test_avec_xg": c_te_a,
            "meilleur_test_sans_xg": best_te_s,
            "meilleur_test_avec_xg": best_te_a,
            "valeur_app": poids_app,
            "vs_marche_pur": poids_vs_marche,
        },
        "apport_xg": apport,
        "brier": metrics_a["brier"],
        "logloss": metrics_a["logloss"],
        "brier_sans_xg": metrics_s["brier"]["modele"],
        "logloss_sans_xg": metrics_s["logloss"]["modele"],
        "n_matchs_compares": metrics_a["n_matchs_compares"],
        "strategie": strategie,
        "duree_s": round(time.time() - t0, 1),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    _resume(res)
    print(f"\nRésultats → {RESULTS_PATH}")


def _compare_poids(test, poids_app):
    """
    Chaque poids < 1 bat-il le marché pur, match par match ?

    C'est LA question des championnats secondaires. Un minimum de
    courbe ne suffit pas : il faut savoir si l'écart survit à sa
    propre incertitude.
    """

    import math

    out = {}
    for w in (0.70, 0.80, 0.90, 0.95, 0.98):
        diffs = []
        for m in test:
            o = m["outcome"]
            pw = max(bt.model_probs(m, w)[o], 1e-12)
            p1 = max(bt.model_probs(m, 1.0)[o], 1e-12)
            diffs.append(math.log(pw) - math.log(p1))
        n = len(diffs)
        if n < 2:
            continue
        moy = sum(diffs) / n
        var = sum((d - moy) ** 2 for d in diffs) / (n - 1)
        et = math.sqrt(var / n) if var > 0 else 0.0
        t = (moy / et) if et > 0 else 0.0
        out[f"{w:.2f}"] = {
            "ecart_moyen": round(moy, 7),
            "t": round(t, 3),
            "bat_le_marche": bool(t > 1.96),
            "pire_que_le_marche": bool(t < -1.96),
        }
    return out


def _resume(r):
    print("\n" + "═" * 64)
    print(" BACKTEST — CHAMPIONNATS SECONDAIRES")
    print("═" * 64)
    print(f"Ligues : {', '.join(r['ligues'])}")
    print(f"Matchs : {r['n_matchs_train']} train / "
          f"{r['n_matchs_test']} test")

    pm = r["poids_marche"]
    print("\nLog-loss par poids marché (plus bas = mieux)")
    print(f"  {'poids':>6s} {'train sans':>11s} {'train avec':>11s} "
          f"{'TEST sans':>11s} {'TEST avec':>11s}")
    for w in pm["courbe_train_avec_xg"]:
        print(f"  {w:>6s} {pm['courbe_train_sans_xg'][w]:11.5f} "
              f"{pm['courbe_train_avec_xg'][w]:11.5f} "
              f"{pm['courbe_test_sans_xg'][w]:11.5f} "
              f"{pm['courbe_test_avec_xg'][w]:11.5f}")
    print(f"  Meilleur HORS ÉCHANTILLON — sans xG "
          f"{pm['meilleur_test_sans_xg']:.2f} · avec xG "
          f"{pm['meilleur_test_avec_xg']:.2f}")

    print("\nChaque poids bat-il le marché pur ? (test apparié)")
    for w, d in pm["vs_marche_pur"].items():
        if d["bat_le_marche"]:
            v = "BAT le marché"
        elif d["pire_que_le_marche"]:
            v = "PIRE que le marché"
        else:
            v = "indiscernable"
        print(f"  w={w} : écart {d['ecart_moyen']:+.6f}  "
              f"t={d['t']:+.2f}  → {v}")

    a = r["apport_xg"]
    if a:
        v = ("SIGNIFICATIF" if a["significatif"]
             else "non significatif")
        print(f"\nApport du xG à w={pm['valeur_app']} "
              f"({a['n_matchs']} matchs) :")
        print(f"  écart {a['ecart_moyen']:+.6f} ± {a['erreur_type']:.6f}"
              f"  t={a['t']:+.2f} → {v}")
    print(f"  log-loss sans xG {r['logloss_sans_xg']:.5f} → "
          f"avec xG {r['logloss']['modele']:.5f}")

    print(f"\nModèle vs marché ({r['n_matchs_compares']} matchs) :")
    print(f"  Brier   modèle {r['brier']['modele']:.5f} · "
          f"B365 {r['brier']['marche_b365']:.5f} · "
          f"clôture {r['brier']['marche_cloture']:.5f}")
    print(f"  logloss modèle {r['logloss']['modele']:.5f} · "
          f"B365 {r['logloss']['marche_b365']:.5f} · "
          f"clôture {r['logloss']['marche_cloture']:.5f}")

    s = r["strategie"]
    print(f"\nStratégie value : {s.get('n_paris')} paris, "
          f"ROI {s.get('roi_pct')}%, CLV moyen {s.get('clv_moyen_pct')}%")


if __name__ == "__main__":
    main()
