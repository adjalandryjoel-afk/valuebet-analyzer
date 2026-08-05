"""
═══════════════════════════════════════════════════════════════
 BACKTEST — marchés PAR ÉQUIPE, PAR MI-TEMPS et TIRS CADRÉS
═══════════════════════════════════════════════════════════════

Les derniers marchés que l'app propose et qui n'avaient jamais été
mesurés : buts par équipe, buts par mi-temps (match et par équipe),
tirs cadrés par équipe.

CE QUE CE BACKTEST PEUT ET NE PEUT PAS DIRE — la distinction est
capitale et ne doit pas être gommée :

  IL NE PEUT PAS dire si le modèle bat le bookmaker. Aucune cote
  historique n'existe pour ces marchés : football-data ne les cote
  pas, et l'endpoint /odds d'API-Football ne renvoie rien sur les
  matchs passés (vérifié sur trois saisons).

  IL PEUT dire si le modèle est CALIBRÉ — et c'est précisément le
  défaut qui compte ici. Si le modèle annonce « 45 % » sur une
  situation qui se produit 38 % du temps, alors CHAQUE pari qu'il
  propose sur ce marché est systématiquement surévalué, et la
  « value » affichée est un artefact de son propre biais. L'app a
  déjà produit ce type de faux positifs : +28 % de value fantôme sur
  les tirs cadrés avant l'ancrage au marché, +19 % sur les mi-temps
  avant le correctif de part de mi-temps. Un marché mal calibré est
  dangereux même sans connaître les cotes.

Les issues, elles, sont toutes observables dans football-data :
FTHG/FTAG (buts par équipe), HTHG/HTAG (mi-temps, dont la seconde par
soustraction) et HST/AST (tirs cadrés).

TIRS CADRÉS — les statistiques de tirs sont reconstruites en
walk-forward, comme les buts. Sans elles, _sot_lambdas renvoie None
et le modèle retombe sur l'approximation buts × SOT_PER_GOAL, repli
que l'app REFUSE de parier : le mesurer reviendrait à tester un
chemin qu'elle n'emprunte jamais.

Sortie : data/backtest_marches_equipe.json
"""

import os
import sys
import json
import math
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util                                    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bt_base", os.path.join(ROOT, "scripts",
                            "backtest_football_data.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

from config import PoissonConfig, SUPPORTED_LEAGUES        # noqa: E402
from modules.poisson_model import PoissonPredictor         # noqa: E402

RESULTS_PATH = os.path.join(ROOT, "data",
                            "backtest_marches_equipe.json")

LIGNES = ("0_5", "1_5", "2_5")
LIGNES_SOT = ("1_5", "2_5", "3_5", "4_5", "5_5")


def _lams(match, weight):
    st, mk = match["stats_lams"], match["market_lams"]
    lam_h = weight * mk[0] + (1 - weight) * st[0]
    lam_a = weight * mk[1] + (1 - weight) * st[1]
    return (max(PoissonConfig.MIN_LAMBDA,
                min(PoissonConfig.MAX_LAMBDA, lam_h)),
            max(PoissonConfig.MIN_LAMBDA,
                min(PoissonConfig.MAX_LAMBDA, lam_a)))


def predictions(match, weight):
    """
    {nom_du_marché: {ligne: probabilité}} — exactement les formules
    de PoissonPredictor._enrich (totaux par équipe, mi-temps, tirs
    cadrés), reproduites ici pour rester au plus près de l'app.
    """

    lam_h, lam_a = _lams(match, weight)
    share = match.get("first_half_share") or PoissonConfig.FIRST_HALF_SHARE
    lam_tot = lam_h + lam_a
    over = PoissonPredictor._over_probs

    out = {
        "buts_domicile": over(lam_h, LIGNES),
        "buts_exterieur": over(lam_a, LIGNES),
        "buts_1re_mi_temps": over(lam_tot * share, LIGNES),
        "buts_2e_mi_temps": over(lam_tot * (1 - share), LIGNES),
        "1re_mt_domicile": over(lam_h * share, LIGNES),
        "1re_mt_exterieur": over(lam_a * share, LIGNES),
        "2e_mt_domicile": over(lam_h * (1 - share), LIGNES),
        "2e_mt_exterieur": over(lam_a * (1 - share), LIGNES),
    }

    sot = match.get("sot_lams")
    if sot:
        out["tirs_cadres_domicile"] = over(sot[0], LIGNES_SOT)
        out["tirs_cadres_exterieur"] = over(sot[1], LIGNES_SOT)
    return out


def issues(match):
    """{nom_du_marché: valeur observée} (nombre de buts / tirs)."""

    gh, ga = match["buts"]
    hth, hta = match.get("mi_temps") or (None, None)
    hst, ast = match.get("sot") or (None, None)

    out = {
        "buts_domicile": gh,
        "buts_exterieur": ga,
    }
    if hth is not None and hta is not None:
        out["buts_1re_mi_temps"] = hth + hta
        out["buts_2e_mi_temps"] = (gh + ga) - (hth + hta)
        out["1re_mt_domicile"] = hth
        out["1re_mt_exterieur"] = hta
        out["2e_mt_domicile"] = gh - hth
        out["2e_mt_exterieur"] = ga - hta
    if hst is not None:
        out["tirs_cadres_domicile"] = hst
    if ast is not None:
        out["tirs_cadres_exterieur"] = ast
    return out


def _seuil(ligne):
    return int(float(ligne.replace("_", "."))) + 1


def mesurer(test, weight):
    """
    Pour chaque marché et chaque ligne : probabilité moyenne annoncée
    contre fréquence réellement observée, avec l'incertitude.

    Le biais (annoncé − observé) est le chiffre qui compte : positif,
    le modèle surestime l'événement et fabriquera de la value là où
    il n'y en a pas.
    """

    accum = {}
    for m in test:
        pred = predictions(m, weight)
        obs = issues(m)
        for marche, lignes in pred.items():
            if marche not in obs or obs[marche] is None:
                continue
            reel = obs[marche]
            if reel < 0:                      # score mi-temps aberrant
                continue
            for ligne, p in lignes.items():
                y = 1 if reel >= _seuil(ligne) else 0
                d = accum.setdefault((marche, ligne),
                                     {"p": [], "y": []})
                d["p"].append(p)
                d["y"].append(y)

    sortie = {}
    for (marche, ligne), d in sorted(accum.items()):
        n = len(d["y"])
        if n < 100:
            continue
        p_moy = sum(d["p"]) / n
        obs_moy = sum(d["y"]) / n
        biais = p_moy - obs_moy
        # Erreur type de la fréquence observée (binomiale)
        et = math.sqrt(max(obs_moy * (1 - obs_moy), 1e-9) / n)
        brier = sum((p - y) ** 2 for p, y in zip(d["p"], d["y"])) / n
        base = sum((obs_moy - y) ** 2 for y in d["y"]) / n
        sortie.setdefault(marche, {})[ligne] = {
            "n": n,
            "annonce": round(p_moy, 4),
            "observe": round(obs_moy, 4),
            "biais": round(biais, 4),
            "biais_en_ecarts_types": round(biais / et, 2) if et else None,
            "significatif": bool(abs(biais / et) > 1.96) if et else None,
            "brier": round(brier, 5),
            "brier_taux_de_base": round(base, 5),
            "bat_le_taux_de_base": bool(brier < base),
        }
    return sortie


def analyser(nom, frames, xg, saisons_test, poids):
    print(f"\n--- {nom} ---")
    matches = bt.prepare_matches(frames, xg, avec_xg=True, avec_sot=True)
    test = [m for m in matches if m["season"] in saisons_test]

    # λ des tirs cadrés : recalculés par la vraie méthode de l'app,
    # sur le contexte reconstruit. Ne concerne que les matchs où les
    # DEUX équipes ont des statistiques de tirs.
    n_sot = sum(1 for m in test if m.get("sot_lams"))
    print(f"    {len(test)} matchs de test, {n_sot} avec des tirs "
          f"cadrés exploitables")
    return {"n_test": len(test), "n_avec_sot": n_sot,
            "marches": mesurer(test, poids)}


def main():
    t0 = time.time()
    poids = PoissonConfig.MARKET_WEIGHT
    res = {"genere_le": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "poids_marche": poids,
        "avertissement": (
            "Aucune cote historique n'existe pour ces marchés. Ce "
            "backtest mesure la CALIBRATION du modèle, pas son edge. "
            "Un biais positif signifie que le modèle surestime "
            "l'événement et fabriquera de la value là où il n'y en a "
            "pas.")}

    print("═══ TOP 5 ═══")
    frames = bt.download_all()
    xg = bt.charger_xg(frames)
    res["top5"] = analyser("top 5", frames, xg, bt.SEASONS_TEST, poids)

    print("\n═══ SECONDAIRES ═══")
    _s = importlib.util.spec_from_file_location(
        "bs", os.path.join(ROOT, "scripts", "backtest_secondaires.py"))
    bs = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(bs)
    bt.DIVISIONS = bs.DIVISIONS_SEC
    bt.SEASONS_ALL = bs.SEASONS_ALL
    bt.SEASONS_BACKTEST = bs.SEASONS_BACKTEST
    frames2 = bt.download_all()
    xg2 = bs.charger_xg_api(frames2)
    res["secondaires"] = analyser("secondaires", frames2, xg2,
                                  bs.SEASONS_TEST, poids)

    res["duree_s"] = round(time.time() - t0, 1)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    _resume(res)
    print(f"\nRésultats → {RESULTS_PATH}")


def _resume(r):
    for cle, titre in (("top5", "LES 5 GRANDS"),
                       ("secondaires", "CHAMPIONNATS SECONDAIRES")):
        d = r[cle]
        print("\n" + "═" * 78)
        print(f" {titre} — {d['n_test']} matchs de test")
        print("═" * 78)
        print(f"  {'marché':22s} {'ligne':6s} {'n':>5s} "
              f"{'annoncé':>8s} {'observé':>8s} {'biais':>8s} "
              f"{'σ':>6s}  verdict")
        print("  " + "-" * 74)
        for marche, lignes in d["marches"].items():
            for ligne, v in lignes.items():
                if v["significatif"]:
                    verdict = ("SURESTIME" if v["biais"] > 0
                               else "sous-estime")
                else:
                    verdict = "calibré"
                if not v["bat_le_taux_de_base"]:
                    verdict += " · pire que le taux de base"
                print(f"  {marche:22s} {ligne:6s} {v['n']:5d} "
                      f"{v['annonce']:8.3f} {v['observe']:8.3f} "
                      f"{v['biais']:+8.3f} "
                      f"{v['biais_en_ecarts_types']:+6.1f}  {verdict}")


if __name__ == "__main__":
    main()
