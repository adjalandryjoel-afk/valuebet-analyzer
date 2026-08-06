"""
═══════════════════════════════════════════════════════
 ROBOT CLV — capture automatique des cotes de clôture
═══════════════════════════════════════════════════════

Lancé toutes les 20 minutes par le Planificateur de tâches
Windows (tâche « ValueBet CLV Auto »). À chaque passage :

  1. S'il n'y a aucun pari en attente sans CLV → sort
     immédiatement (zéro requête, zéro crédit).
  2. Repère l'heure de coup d'envoi de chaque match via
     l'endpoint /events de The Odds API — GRATUIT (0 crédit).
     Les heures trouvées sont mises en cache ; les matchs
     introuvables (ligues locales non couvertes) ne sont
     re-cherchés que toutes les 6 h.
  3. Si un match démarre dans ≤ 22 minutes : capture les
     cotes de clôture (ClvTracker, ~2 crédits, uniquement
     la ligue concernée) et enregistre le CLV en base.
  4. Garde-fou : si le quota mensuel restant < 60 crédits,
     aucune capture payante n'est lancée.

Journal : data/clv_daemon.log — état : data/clv_daemon_state.json
"""

import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone

# ── Ancrage au projet (le Planificateur ne fixe pas le cwd) ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

LOG_PATH = os.path.join(ROOT, "data", "clv_daemon.log")
STATE_PATH = os.path.join(ROOT, "data", "clv_daemon_state.json")

# Fenêtre de capture autour du coup d'envoi (minutes)
WINDOW_BEFORE = 22   # cadence 20 min → un passage garanti dedans
WINDOW_AFTER = 10
# Re-chercher un match introuvable au plus toutes les X heures
LOOKUP_COOLDOWN_H = 6
# Ne jamais capturer si le quota mensuel passe sous ce seuil
MIN_QUOTA = 60


def _open_log():
    """Journal en append UTF-8, tronqué s'il dépasse 300 Ko."""
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 300_000:
            with io.open(LOG_PATH, encoding="utf-8", errors="replace") as f:
                keep = f.read()[-100_000:]
            with io.open(LOG_PATH, "w", encoding="utf-8") as f:
                f.write("[...journal tronqué...]\n" + keep)
    except OSError:
        pass
    return io.open(LOG_PATH, "a", encoding="utf-8", errors="replace")


def main():
    log = _open_log()
    # Tout print (y compris ceux de ClvTracker) part dans le journal
    sys.stdout = sys.stderr = log

    def say(msg):
        print(f"{datetime.now():%Y-%m-%d %H:%M} {msg}", flush=True)

    try:
        from config import APIKeys
        from modules.database_manager import DatabaseManager
        from modules.clv_tracker import ClvTracker
        from modules.odds_collector import OddsAPICollector
        import requests
        from rapidfuzz import fuzz
    except Exception as e:
        say(f"ERREUR imports : {e}")
        return

    db = DatabaseManager()

    # Miroir Supabase : rapatrie les paris faits depuis le
    # téléphone pour leur capturer aussi la cote de clôture
    try:
        bilan_sync = db.hydrate_from_cloud()
        if any(bilan_sync.values()):
            say(f"Sync cloud : {bilan_sync}")
    except Exception as e:
        say(f"Sync cloud impossible : {e}")

    # Critère : la cote de clôture manque. PAS « le résultat manque »
    # — un pari réglé le soir même sortait du périmètre avant que le
    # robot ait eu sa fenêtre, et son CLV était perdu pour toujours.
    pending = [b for b in db.get_bets_awaiting_clv()
               if not b.get("closing_odds")]
    if not pending:
        # Battement de cœur : sans cette ligne, un passage à vide est
        # indiscernable d'un robot qui ne tourne pas. Cas déjà vécu —
        # un coup d'envoi repéré, sa fenêtre passée, et aucune trace
        # ni de capture ni d'échec dans le journal.
        say("Aucun pari en attente de CLV — rien à faire.")
        log.close()
        return

    # ── État persistant ──
    state = {"kickoffs": {}, "lookups": {}, "missed": []}
    try:
        with io.open(STATE_PATH, encoding="utf-8") as f:
            state.update(json.load(f))
    except (OSError, ValueError):
        pass

    now = datetime.now(timezone.utc)

    # Purge des entrées de plus de 7 jours
    def _fresh(iso):
        try:
            return datetime.fromisoformat(iso) > now - timedelta(days=7)
        except ValueError:
            return False

    state["kickoffs"] = {k: v for k, v in state["kickoffs"].items()
                         if _fresh(v.get("utc", ""))}
    state["lookups"] = {k: v for k, v in state["lookups"].items()
                        if _fresh(v)}
    state["missed"] = [m for m in state["missed"]
                       if m in state["kickoffs"]]

    # ── Regrouper les paris par match ──
    groupes = {}
    for b in pending:
        key = f"{(b.get('home_team') or '').strip()}|" \
              f"{(b.get('away_team') or '').strip()}"
        groupes.setdefault(key, []).append(b)

    # ── Découverte des coups d'envoi (endpoint /events, GRATUIT) ──
    a_chercher = []
    for key in groupes:
        cached = state["kickoffs"].get(key)
        if cached:
            continue
        last = state["lookups"].get(key)
        if last and _fresh(last) and \
                datetime.fromisoformat(last) > now - timedelta(
                    hours=LOOKUP_COOLDOWN_H):
            continue
        a_chercher.append(key)

    # Quota : repris du dernier passage s'il n'est pas rafraîchi ici.
    #
    # Il n'était renseigné que dans la branche `if a_chercher:`. Or une
    # affiche n'entre dans la fenêtre de capture que si son coup
    # d'envoi est DÉJÀ en cache — et une affiche dont le coup d'envoi
    # est en cache est justement sautée de `a_chercher`. Au moment de
    # la capture payante, `quota` valait donc typiquement None et le
    # garde-fou `int(quota) < MIN_QUOTA` était SAUTÉ. Mesuré à quota
    # simulé de 3 crédits : capture annulée si l'état est vide, mais
    # 2 crédits dépensés dès que le coup d'envoi est en cache.
    quota = state.get("quota")
    if a_chercher:
        events_par_ligue = {}
        for ligue, sport_key in OddsAPICollector.LEAGUE_KEYS.items():
            if sport_key in events_par_ligue:
                continue
            try:
                r = requests.get(
                    "https://api.the-odds-api.com/v4/sports/"
                    f"{sport_key}/events",
                    params={"apiKey": APIKeys.ODDS_API_KEY},
                    timeout=20,
                )
                q = r.headers.get("x-requests-remaining")
                if q is not None:
                    quota = q
                    state["quota"] = q
                    state["quota_ts"] = now.isoformat()
                events_par_ligue[sport_key] = (
                    r.json() if r.status_code == 200 else []
                )
            except Exception:
                events_par_ligue[sport_key] = []

        for key in a_chercher:
            home, away = key.split("|", 1)
            state["lookups"][key] = now.isoformat()
            trouve = None
            for ligue, sport_key in OddsAPICollector.LEAGUE_KEYS.items():
                for ev in events_par_ligue.get(sport_key, []):
                    sh = fuzz.token_sort_ratio(
                        home.lower(), str(ev.get("home_team", "")).lower())
                    sa = fuzz.token_sort_ratio(
                        away.lower(), str(ev.get("away_team", "")).lower())
                    if sh >= 65 and sa >= 65 and ev.get("commence_time"):
                        trouve = (ligue, ev["commence_time"])
                        break
                if trouve:
                    break
            if trouve:
                state["kickoffs"][key] = {
                    "ligue": trouve[0], "utc": trouve[1]}
                say(f"Coup d'envoi repéré : {key.replace('|', ' vs ')} "
                    f"→ {trouve[1]} ({trouve[0]})")
            else:
                # Un échec de recherche doit se VOIR. Sans cette
                # ligne, l'affiche s'enterrait dans state["lookups"]
                # et on ne pouvait pas distinguer « The Odds API ne
                # couvre pas ce match » de « le robot n'a pas tourné ».
                # Mesuré : 12 recherches, 2 coups d'envoi trouvés,
                # 10 échecs totalement muets.
                # On dit ce qu'on OBSERVE, pas ce qu'on suppose : une
                # affiche peut être absente parce que la compétition
                # n'est pas couverte, mais aussi parce que le match
                # est trop lointain, déjà joué, ou que les noms
                # diffèrent. Affirmer « non couverte » était faux
                # pour des ligues qui le sont.
                say(f"Affiche introuvable chez The Odds API : "
                    f"{key.replace('|', ' vs ')} — pas de CLV pour "
                    f"l'instant (compétition non couverte, match trop "
                    f"lointain, ou noms d'équipes différents)")

    # ── Matchs dans la fenêtre de capture ──
    dus, ligues_dues = [], {}
    for key, bets in groupes.items():
        info = state["kickoffs"].get(key)
        if not info:
            continue
        try:
            ko = datetime.fromisoformat(
                info["utc"].replace("Z", "+00:00"))
        except ValueError:
            continue
        delta_min = (ko - now).total_seconds() / 60.0
        if -WINDOW_AFTER <= delta_min <= WINDOW_BEFORE:
            dus.extend(bets)
            ligue = info.get("ligue")
            if ligue in OddsAPICollector.LEAGUE_KEYS:
                ligues_dues[ligue] = OddsAPICollector.LEAGUE_KEYS[ligue]
        elif delta_min < -WINDOW_AFTER and key not in state["missed"]:
            state["missed"].append(key)
            say(f"CLV manqué (PC éteint au coup d'envoi ?) : "
                f"{key.replace('|', ' vs ')}")

    if dus:
        # Défaut INVERSÉ : sans quota connu et frais, on ne dépense
        # rien. Un garde-fou qui s'efface quand l'information manque
        # ne garde rien du tout.
        def _quota_frais():
            try:
                qts = datetime.fromisoformat(state.get("quota_ts", ""))
                return (now - qts) < timedelta(hours=24)
            except (ValueError, TypeError):
                return False

        # Rafraîchissement GRATUIT avant toute dépense. /events est
        # facturé 0 crédit — mesuré, en-tête x-requests-last = 0. Sans
        # ce rappel, le quota pouvait rester inconnu pendant des jours
        # (les recherches de coup d'envoi ont 6 h de refroidissement)
        # et le garde-fou, désormais strict, aurait bloqué toutes les
        # captures indéfiniment.
        if quota is None or not _quota_frais():
            try:
                r = requests.get(
                    "https://api.the-odds-api.com/v4/sports/"
                    "soccer_epl/events",
                    params={"apiKey": APIKeys.ODDS_API_KEY}, timeout=20)
                q = r.headers.get("x-requests-remaining")
                if q is not None:
                    quota = q
                    state["quota"] = q
                    state["quota_ts"] = now.isoformat()
            except Exception:
                pass

        if quota is None or not _quota_frais():
            say("Quota The Odds API inconnu ou périmé — capture "
                "annulée par prudence (aucune requête émise). Il sera "
                "rafraîchi au prochain repérage de coup d'envoi.")
        elif int(quota) < MIN_QUOTA:
            say(f"Quota trop bas ({quota} crédits) — capture annulée")
        elif not ligues_dues:
            # GARDE-FOU DE QUOTA. Sans lui, un match dont la
            # compétition n'est pas reconnue laissait le tracker avec
            # TOUTES ses clés : un balayage complet, ~2 crédits ×
            # le nombre de championnats suivis, en silence, toutes
            # les 20 minutes. Mesuré à 56 crédits par passage — et
            # l'élargissement de la couverture le porterait à près
            # de 100, un cinquième du quota mensuel d'un coup.
            #
            # Un match introuvable n'est pas une raison de tout
            # scanner : c'est une raison de le dire et de passer.
            affiches = sorted({
                f"{b.get('home_team')} vs {b.get('away_team')}"
                for b in dus})
            say(f"{len(dus)} pari(s) dus mais aucune compétition "
                f"reconnue parmi les clés suivies — capture annulée, "
                f"AUCUNE requête émise. Affiches : "
                f"{', '.join(affiches[:5])}")
        else:
            tracker = ClvTracker(db)
            tracker.LEAGUE_KEYS = ligues_dues  # ligues ciblées
            res = tracker.capture_closing_odds(dus)
            say(f"Capture : {res['captures']} CLV enregistré(s), "
                f"{res['ignores']} ignoré(s), "
                f"{res['erreurs']} erreur(s) — "
                f"quota restant {res['quota_restant']}")

    if not dus:
        # Deuxième battement de cœur : il y a des paris en attente,
        # mais aucun match dans la fenêtre de capture. Sans cette
        # ligne, un passage entier ne laissait AUCUNE trace et on ne
        # pouvait pas distinguer « rien à capturer maintenant » de
        # « le robot n'a pas tourné ».
        say(f"{len(pending)} pari(s) en attente, aucun match dans la "
            f"fenêtre de capture — rien à faire "
            f"(quota connu : {quota or 'inconnu'})")

    # ── Sauvegarde de l'état ──
    try:
        with io.open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
    except OSError as e:
        say(f"ERREUR sauvegarde état : {e}")

    log.close()


if __name__ == "__main__":
    main()
