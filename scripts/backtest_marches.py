"""
═══════════════════════════════════════════════════════════════
 BACKTEST — MARCHÉS SECONDAIRES : Over/Under 2.5 et BTTS
═══════════════════════════════════════════════════════════════

Le verdict établi (« aucun edge ») ne portait que sur le 1X2. Il
restait à tester les marchés que l'app propose aussi, et sur lesquels
un edge est souvent supposé plus accessible.

DEUX MARCHÉS, DEUX NIVEAUX DE PREUVE — la différence est essentielle
et ne doit jamais être gommée dans la lecture des résultats :

  • OVER/UNDER 2.5 — TESTABLE À FOND. football-data fournit la cote
    d'ouverture (B365) ET la cote de clôture (Pinnacle). On peut donc
    mesurer exactement comme pour le 1X2 : le modèle bat-il le marché,
    et le CLV est-il positif ?

  • BTTS — TESTABLE SEULEMENT À MOITIÉ. Aucune source historique de
    cotes BTTS n'existe : football-data n'a pas la colonne (vérifié,
    0 sur 120 colonnes) et l'endpoint /odds d'API-Football ne renvoie
    RIEN sur les matchs passés (vérifié sur trois saisons de Premier
    League : 0 bloc de cotes). On ne peut donc PAS dire si le modèle
    bat le bookmaker sur le BTTS.
    Ce qui reste mesurable, et qui a du sens : la composante
    statistique améliore-t-elle la prédiction du BTTS par rapport à
    un λ purement dérivé du marché ? Et le modèle est-il calibré ?
    Un modèle mal calibré sur le BTTS ne peut pas y avoir d'edge ;
    bien calibré, il en a peut-être un — mais ce backtest ne pourra
    pas le prouver.

Sortie : data/backtest_marches.json
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

from config import PoissonConfig, ValueBetConfig, KellyConfig  # noqa: E402
from modules.poisson_model import PoissonPredictor            # noqa: E402
from modules.odds_utils import novig_probs, margin_ok         # noqa: E402

RESULTS_PATH = os.path.join(ROOT, "data", "backtest_marches.json")


# ══════════════════════════════════════════════════════════════
#  PROBABILITÉS DES MARCHÉS SECONDAIRES
# ══════════════════════════════════════════════════════════════

def probs_marches(match, weight):
    """
    (p_over25, p_btts) depuis la MÊME matrice Dixon-Coles que le 1X2.

    On reconstruit la matrice plutôt que d'appeler model_probs :
    celui-ci ne renvoie pas le BTTS, et refaire le calcul ici garantit
    que les deux marchés sortent exactement du même λ — sinon on
    comparerait deux modèles différents sans le savoir.
    """

    st, mk = match["stats_lams"], match["market_lams"]
    lam_h = weight * mk[0] + (1 - weight) * st[0]
    lam_a = weight * mk[1] + (1 - weight) * st[1]
    lam_h = max(PoissonConfig.MIN_LAMBDA,
                min(PoissonConfig.MAX_LAMBDA, lam_h))
    lam_a = max(PoissonConfig.MIN_LAMBDA,
                min(PoissonConfig.MAX_LAMBDA, lam_a))

    matrix = PoissonPredictor._score_matrix(lam_h, lam_a)
    over25 = btts = total = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            total += p
            if i + j >= 3:
                over25 += p
            if i >= 1 and j >= 1:
                btts += p
    if total > 0:
        over25 /= total
        btts /= total
    return over25, btts


def novig_2(o_pour, o_contre):
    """Probabilité no-vig d'un marché à deux issues (Shin)."""

    if not (o_pour > 1 and o_contre > 1):
        return None
    p = novig_probs([o_pour, o_contre])
    return p[0] if p else None


# ══════════════════════════════════════════════════════════════
#  MESURES
# ══════════════════════════════════════════════════════════════

def brier_bin(probs, issues):
    return sum((p - y) ** 2 for p, y in zip(probs, issues)) / len(issues)


def logloss_bin(probs, issues):
    t = 0.0
    for p, y in zip(probs, issues):
        p = min(max(p, 1e-12), 1 - 1e-12)
        t -= math.log(p) if y else math.log(1 - p)
    return t / len(issues)


def apparie_bin(pa, pb, issues):
    """
    pb bat-il pa, match par match ? (positif = pb meilleur)

    Test apparié : les deux séries portent sur les MÊMES matchs, ce
    qui élimine la variance du tirage. Un écart moyen ne veut rien
    dire sans son incertitude.
    """

    d = []
    for a, b, y in zip(pa, pb, issues):
        a = min(max(a, 1e-12), 1 - 1e-12)
        b = min(max(b, 1e-12), 1 - 1e-12)
        la = math.log(a) if y else math.log(1 - a)
        lb = math.log(b) if y else math.log(1 - b)
        d.append(lb - la)
    n = len(d)
    if n < 2:
        return None
    moy = sum(d) / n
    var = sum((v - moy) ** 2 for v in d) / (n - 1)
    et = math.sqrt(var / n) if var > 0 else 0.0
    t = (moy / et) if et > 0 else 0.0
    return {"n": n, "ecart_moyen": round(moy, 7),
            "erreur_type": round(et, 7), "t": round(t, 3),
            "significatif": bool(abs(t) > 1.96)}


def calibration(probs, issues, n_paniers=10):
    """Probabilité prédite contre fréquence observée, par décile."""

    paniers = []
    for k in range(n_paniers):
        lo, hi = k / n_paniers, (k + 1) / n_paniers
        idx = [i for i, p in enumerate(probs)
               if (lo <= p < hi) or (k == n_paniers - 1 and p == 1.0)]
        if len(idx) < 20:
            continue
        paniers.append({
            "tranche": f"{lo:.1f}-{hi:.1f}",
            "n": len(idx),
            "predite": round(sum(probs[i] for i in idx) / len(idx), 4),
            "observee": round(sum(issues[i] for i in idx) / len(idx), 4),
        })
    return paniers


def strategie_ou(test, poids):
    """
    Stratégie value sur l'Over/Under, quart de Kelly.

    Le CLV se mesure contre la CLÔTURE Pinnacle : c'est l'indicateur
    avancé. Un CLV négatif dit que le pari était déjà mauvais au
    moment où il a été pris, indépendamment du résultat.
    """

    bankroll = KellyConfig.DEFAULT_BANKROLL
    plafond = bankroll * KellyConfig.MAX_STAKE_PERCENTAGE / 100
    paris, profit, clvs = [], 0.0, []
    mise_totale = 0.0

    for m in test:
        p_ov, _ = probs_marches(m, poids)
        o_ov, o_un = m.get("ou_open") or (0, 0)
        if not (o_ov > 1 and o_un > 1):
            continue
        # Même garde-fou que l'app : une paire de cotes à marge
        # aberrante ne doit servir d'ancre à aucun calcul. 0.20 est
        # le seuil des marchés à deux issues (value_detector).
        if not margin_ok([o_ov, o_un], max_margin=0.20):
            continue

        for cote, p, gagne in ((o_ov, p_ov, m["over25"] == 1),
                               (o_un, 1 - p_ov, m["over25"] == 0)):
            if cote > ValueBetConfig.MAX_ODDS:
                continue
            value = p * cote - 1
            if value < bt.value_threshold(cote):
                continue
            b = cote - 1
            kelly = (p * b - (1 - p)) / b
            mise = min(max(kelly * KellyConfig.KELLY_FRACTION, 0)
                       * bankroll, plafond)
            if mise <= 0:
                continue
            gain = mise * b if gagne else -mise
            profit += gain
            mise_totale += mise
            paris.append({"cote": cote, "value": value, "gagne": gagne,
                          "mise": mise})

            # CLV : cote prise contre cote de clôture, en probabilité
            c_ov, c_un = m.get("ou_close") or (0, 0)
            p_clo = novig_2(c_ov, c_un) if cote == o_ov else \
                novig_2(c_un, c_ov)
            if p_clo:
                mise_impl = 1 / cote
                clvs.append((p_clo / mise_impl - 1) * 100)

    n = len(paris)
    return {
        "n_paris": n,
        # ROI = profit / TOTAL RÉELLEMENT MISÉ. La première version
        # divisait par une mise nominale de 1 % de bankroll par pari,
        # alors que les mises sont en quart de Kelly plafonné à 2 % :
        # le ROI en sortait gonflé (−27 % au lieu de −15 %, +34 % au
        # lieu de +19 %). Même convention que simulate_strategy.
        "roi_pct": round(100 * profit / mise_totale, 2)
        if mise_totale else None,
        "mise_totale": round(mise_totale),
        "profit": round(profit),
        "taux_reussite": round(100 * sum(p["gagne"] for p in paris) / n, 1)
        if n else None,
        "cote_moyenne": round(sum(p["cote"] for p in paris) / n, 2)
        if n else None,
        "clv_moyen_pct": round(sum(clvs) / len(clvs), 2) if clvs else None,
        "clv_positif_pct": round(100 * sum(1 for c in clvs if c > 0)
                                 / len(clvs), 1) if clvs else None,
        "n_avec_clv": len(clvs),
        # Un ROI sur quelques dizaines de paris ne veut rien dire tant
        # qu'on ne connaît pas son incertitude. Ce projet s'est déjà
        # fait prendre par un « +45 % » sur 13 paris qui n'était que
        # du bruit : l'intervalle voyage désormais AVEC le chiffre.
        "roi_ic95": _bootstrap_roi(paris),
    }


def _bootstrap_roi(paris, n_tirages=20000, graine=20260805):
    """IC95 du ROI par rééchantillonnage des paris."""

    import random

    if len(paris) < 5:
        return None
    rng = random.Random(graine)
    n = len(paris)
    rois = []
    for _ in range(n_tirages):
        ech = [paris[rng.randrange(n)] for _ in range(n)]
        m = sum(p["mise"] for p in ech)
        g = sum((p["mise"] * (p["cote"] - 1)) if p["gagne"]
                else -p["mise"] for p in ech)
        rois.append(100 * g / m if m else 0.0)
    rois.sort()
    bas = rois[int(0.025 * n_tirages)]
    haut = rois[int(0.975 * n_tirages)]
    return {
        "bas": round(bas, 1), "haut": round(haut, 1),
        "zero_dedans": bool(bas <= 0 <= haut),
        "part_rejeux_negatifs": round(
            sum(1 for r in rois if r < 0) / n_tirages, 3),
    }


# ══════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════

def analyser(nom, frames, xg, saisons_test, poids_app):
    """Analyse complète d'un périmètre (top 5 ou secondaires)."""

    print(f"\n--- {nom} : reconstruction ---")
    m_sans = bt.prepare_matches(frames, xg, avec_xg=False)
    m_avec = bt.prepare_matches(frames, xg, avec_xg=True)
    te_s = [m for m in m_sans if m["season"] in saisons_test]
    te_a = [m for m in m_avec if m["season"] in saisons_test]
    print(f"    test : {len(te_a)} matchs")

    ou_reel = [m["over25"] for m in te_a]
    btts_reel = [m["btts"] for m in te_a]

    # ── Courbes de poids ──
    courbe_ou, courbe_btts = {}, {}
    for w in bt.MARKET_WEIGHT_GRID:
        po = [probs_marches(m, w)[0] for m in te_a]
        pb = [probs_marches(m, w)[1] for m in te_a]
        courbe_ou[f"{w:.2f}"] = round(logloss_bin(po, ou_reel), 5)
        courbe_btts[f"{w:.2f}"] = round(logloss_bin(pb, btts_reel), 5)

    # ── Over/Under : le modèle contre le marché ──
    p_mod = [probs_marches(m, poids_app)[0] for m in te_a]
    p_pur = [probs_marches(m, 1.0)[0] for m in te_a]
    p_ouv, p_clo = [], []
    for m in te_a:
        o = m.get("ou_open") or (0, 0)
        c = m.get("ou_close") or (0, 0)
        p_ouv.append(novig_2(o[0], o[1]))
        p_clo.append(novig_2(c[0], c[1]))

    idx = [i for i in range(len(te_a))
           if p_ouv[i] is not None and p_clo[i] is not None]
    y = [ou_reel[i] for i in idx]
    mod = [p_mod[i] for i in idx]
    pur = [p_pur[i] for i in idx]
    ouv = [p_ouv[i] for i in idx]
    clo = [p_clo[i] for i in idx]

    ou = {
        "n_compares": len(idx),
        "brier": {
            "modele": round(brier_bin(mod, y), 5),
            "modele_marche_pur": round(brier_bin(pur, y), 5),
            "cote_ouverture": round(brier_bin(ouv, y), 5),
            "cote_cloture": round(brier_bin(clo, y), 5),
        },
        "logloss": {
            "modele": round(logloss_bin(mod, y), 5),
            "modele_marche_pur": round(logloss_bin(pur, y), 5),
            "cote_ouverture": round(logloss_bin(ouv, y), 5),
            "cote_cloture": round(logloss_bin(clo, y), 5),
        },
        "modele_vs_ouverture": apparie_bin(ouv, mod, y),
        "modele_vs_cloture": apparie_bin(clo, mod, y),
        "stats_vs_marche_pur": apparie_bin(pur, mod, y),
        "courbe_logloss_test": courbe_ou,
        "meilleur_poids": float(min(courbe_ou, key=courbe_ou.get)),
        "calibration": calibration(mod, y),
        "strategie": strategie_ou(te_a, poids_app),
    }

    # ── BTTS : pas de cote historique, donc pas de verdict marché ──
    pb_mod = [probs_marches(m, poids_app)[1] for m in te_a]
    pb_pur = [probs_marches(m, 1.0)[1] for m in te_a]
    taux_base = sum(btts_reel) / len(btts_reel)
    btts = {
        "avertissement": ("AUCUNE cote BTTS historique n'existe "
                          "(football-data n'a pas la colonne ; /odds "
                          "d'API-Football ne renvoie rien sur les "
                          "matchs passés). On ne peut donc PAS dire "
                          "si le modèle bat le bookmaker sur ce "
                          "marché — seulement s'il prédit bien."),
        "n": len(te_a),
        "taux_observe": round(taux_base, 4),
        "brier": {
            "modele": round(brier_bin(pb_mod, btts_reel), 5),
            "modele_marche_pur": round(brier_bin(pb_pur, btts_reel), 5),
            "taux_de_base": round(
                brier_bin([taux_base] * len(btts_reel), btts_reel), 5),
        },
        "logloss": {
            "modele": round(logloss_bin(pb_mod, btts_reel), 5),
            "modele_marche_pur": round(logloss_bin(pb_pur, btts_reel), 5),
            "taux_de_base": round(
                logloss_bin([taux_base] * len(btts_reel), btts_reel), 5),
        },
        "stats_vs_marche_pur": apparie_bin(pb_pur, pb_mod, btts_reel),
        "vs_taux_de_base": apparie_bin(
            [taux_base] * len(btts_reel), pb_mod, btts_reel),
        "courbe_logloss_test": courbe_btts,
        "meilleur_poids": float(min(courbe_btts, key=courbe_btts.get)),
        "calibration": calibration(pb_mod, btts_reel),
    }

    # ── Apport du xG sur ces deux marchés ──
    po_s = [probs_marches(m, poids_app)[0] for m in te_s]
    pb_s = [probs_marches(m, poids_app)[1] for m in te_s]
    apport = {
        "over_under": apparie_bin(po_s, p_mod, ou_reel),
        "btts": apparie_bin(pb_s, pb_mod, btts_reel),
    }

    return {"n_test": len(te_a), "over_under": ou, "btts": btts,
            "apport_xg": apport}


def main():
    t0 = time.time()
    poids_app = PoissonConfig.MARKET_WEIGHT
    res = {"genere_le": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "poids_app": poids_app}

    # ── Périmètre 1 : les 5 grands (xG Understat) ──
    print("═══ TOP 5 ═══")
    frames = bt.download_all()
    xg = bt.charger_xg(frames)
    res["top5"] = analyser("top 5", frames, xg, bt.SEASONS_TEST,
                           poids_app)

    # ── Périmètre 2 : secondaires (xG API-Football) ──
    print("\n═══ CHAMPIONNATS SECONDAIRES ═══")
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
                                  bs.SEASONS_TEST, poids_app)

    res["duree_s"] = round(time.time() - t0, 1)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    _resume(res)
    print(f"\nRésultats → {RESULTS_PATH}")


def _resume(r):
    for cle, titre in (("top5", "LES 5 GRANDS"),
                       ("secondaires", "CHAMPIONNATS SECONDAIRES")):
        d = r[cle]
        print("\n" + "═" * 66)
        print(f" {titre} — {d['n_test']} matchs de test")
        print("═" * 66)

        ou = d["over_under"]
        print(f"\nOVER/UNDER 2.5 ({ou['n_compares']} matchs comparés)")
        b, l = ou["brier"], ou["logloss"]
        print(f"  {'':22s} {'Brier':>9s} {'log-loss':>10s}")
        for nom, k in (("modèle (w=0.90)", "modele"),
                       ("modèle marché pur", "modele_marche_pur"),
                       ("cote d'ouverture", "cote_ouverture"),
                       ("cote de clôture", "cote_cloture")):
            print(f"  {nom:22s} {b[k]:9.5f} {l[k]:10.5f}")
        for nom, k in (("contre l'ouverture", "modele_vs_ouverture"),
                       ("contre la clôture", "modele_vs_cloture"),
                       ("stats contre marché pur", "stats_vs_marche_pur")):
            t = ou[k]
            if not t:
                continue
            v = ("BAT" if t["t"] > 1.96 else
                 "PIRE" if t["t"] < -1.96 else "indiscernable")
            print(f"  {nom:24s} écart {t['ecart_moyen']:+.6f}  "
                  f"t={t['t']:+.2f}  → {v}")
        print(f"  meilleur poids hors échantillon : {ou['meilleur_poids']}")
        s = ou["strategie"]
        ic = s.get("roi_ic95")
        txt_ic = ""
        if ic:
            txt_ic = (f" · IC95 [{ic['bas']:+.1f}% ; {ic['haut']:+.1f}%]"
                      f"{' — zéro dedans, ne prouve rien' if ic['zero_dedans'] else ''}")
        print(f"  stratégie : {s['n_paris']} paris · ROI {s['roi_pct']}%"
              f"{txt_ic}")
        print(f"              CLV {s['clv_moyen_pct']}% "
              f"(positif sur {s['clv_positif_pct']}% des "
              f"{s['n_avec_clv']} paris mesurables)")

        bt_ = d["btts"]
        print(f"\nBTTS ({bt_['n']} matchs) — "
              f"observé {bt_['taux_observe']:.1%}")
        print(f"  !! {bt_['avertissement'][:60]}...")
        print(f"  log-loss  modèle {bt_['logloss']['modele']:.5f} · "
              f"marché pur {bt_['logloss']['modele_marche_pur']:.5f} · "
              f"taux de base {bt_['logloss']['taux_de_base']:.5f}")
        for nom, k in (("stats contre marché pur", "stats_vs_marche_pur"),
                       ("modèle contre taux de base", "vs_taux_de_base")):
            t = bt_[k]
            if not t:
                continue
            v = ("MIEUX" if t["t"] > 1.96 else
                 "PIRE" if t["t"] < -1.96 else "indiscernable")
            print(f"  {nom:28s} t={t['t']:+.2f} → {v}")

        a = d["apport_xg"]
        print("\nApport du xG sur ces marchés :")
        for nom, k in (("Over/Under", "over_under"), ("BTTS", "btts")):
            t = a[k]
            if not t:
                continue
            v = "significatif" if t["significatif"] else "non significatif"
            print(f"  {nom:12s} écart {t['ecart_moyen']:+.6f}  "
                  f"t={t['t']:+.2f} → {v}")


if __name__ == "__main__":
    main()
