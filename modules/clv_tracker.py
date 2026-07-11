"""
═══════════════════════════════════════════════════════
 MODULE CLV TRACKER — Closing Line Value
 Capture des cotes de clôture via The Odds API
═══════════════════════════════════════════════════════

Le CLV (Closing Line Value) est le KPI n°1 des parieurs
professionnels : battre régulièrement la cote de clôture
sans marge (la plus efficiente du marché) prouve que le
modèle a un vrai edge, bien avant que le ROI ne soit
statistiquement significatif.

Principe :
  1. On prend les paris en attente (get_pending_bets)
  2. On retrouve chaque match sur The Odds API (fuzzy match
     sur les noms d'équipes, ligue par ligue avec cache
     mémoire → une seule requête API par ligue et par appel)
  3. On extrait la cote de clôture du même marché/sélection
     (Pinnacle en priorité, sinon moyenne des bookmakers)
  4. Cote juste de clôture = 1 / prob no-vig (méthode de
     Shin) sur le marché complet
  5. CLV% = (cote_prise / cote_juste_clôture − 1) × 100

Économie de quota : le plan gratuit est limité à
500 crédits/mois — chaque ligue n'est interrogée qu'une
fois par appel de capture_closing_odds.
"""

import requests
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from config import APIKeys
from modules.odds_utils import novig_probs
from modules.odds_collector import OddsAPICollector


class ClvTracker:
    """
    Traqueur de Closing Line Value.

    Regroupe les paris en attente par match, retrouve les
    cotes de clôture sur The Odds API et enregistre pour
    chaque pari la cote de clôture et le CLV% via
    DatabaseManager.update_bet_closing.

    Marchés supportés :
      • 1X2 (market = "1X2", selection "1"/"X"/"2")
      • Over/Under 2.5 (market contenant "2.5",
        selection Over/Under)
    Les autres marchés sont comptés dans "ignores".
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    # Seuil rapidfuzz (token_sort_ratio) pour matcher un événement
    FUZZY_THRESHOLD = 65

    # Ligues interrogées, dans l'ordre (clés The Odds API)
    LEAGUE_KEYS = OddsAPICollector.LEAGUE_KEYS

    def __init__(self, db=None):
        """
        Args:
            db: instance de DatabaseManager (créée si None)
        """

        if db is None:
            from modules.database_manager import DatabaseManager
            db = DatabaseManager()

        self.db = db
        self.api_key = APIKeys.ODDS_API_KEY

        # Cache mémoire : sport_key → liste d'événements
        # (None = requête déjà tentée et échouée, on ne réessaie pas)
        self._events_cache: Dict[str, Optional[List[Dict]]] = {}

        # Quota restant (header x-requests-remaining)
        self._remaining_requests: Optional[str] = None

    # ─── API PUBLIQUE ────────────────────────────────

    def capture_closing_odds(self, pending_bets: List[Dict]) -> Dict:
        """
        Capture les cotes de clôture des paris en attente et
        enregistre leur CLV en base.

        Args:
            pending_bets: liste de dicts issus de
                DatabaseManager.get_pending_bets() — chaque pari
                doit contenir id, home_team, away_team, market,
                selection, bookmaker_odds.

        Returns:
            {
                "traites":       nb de paris examinés,
                "captures":      nb de CLV enregistrés en base,
                "ignores":       nb ignorés (marché non supporté,
                                 match ou cotes introuvables),
                "erreurs":       nb d'erreurs inattendues,
                "quota_restant": crédits API restants (str ou None),
            }
        """

        resume = {
            "traites": 0,
            "captures": 0,
            "ignores": 0,
            "erreurs": 0,
            "quota_restant": self._remaining_requests,
        }

        if not pending_bets:
            return resume

        # ── Regrouper les paris par match ──
        groupes: Dict[Tuple[str, str], List[Dict]] = {}
        for bet in pending_bets:
            key = (
                str(bet.get("home_team") or "").strip(),
                str(bet.get("away_team") or "").strip(),
            )
            groupes.setdefault(key, []).append(bet)

        # ── Traiter chaque match ──
        for (home, away), bets in groupes.items():
            resume["traites"] += len(bets)

            try:
                event = self._find_event(home, away)
            except Exception as e:
                print(f"   ❌ Erreur recherche {home} - {away}: {e}")
                resume["erreurs"] += len(bets)
                continue

            if not event:
                # Match introuvable sur The Odds API (déjà joué,
                # ligue non couverte…) → paris ignorés
                resume["ignores"] += len(bets)
                continue

            for bet in bets:
                try:
                    statut = self._process_bet(bet, event)
                    resume[statut] += 1
                except Exception as e:
                    print(f"   ❌ Erreur pari #{bet.get('id')}: {e}")
                    resume["erreurs"] += 1

        resume["quota_restant"] = self._remaining_requests
        return resume

    # ─── TRAITEMENT D'UN PARI ────────────────────────

    def _process_bet(self, bet: Dict, event: Dict) -> str:
        """
        Calcule et enregistre le CLV d'un pari sur un événement.

        Returns:
            "captures" si le CLV a été enregistré, "ignores" sinon.
        """

        market = str(bet.get("market") or "")
        selection = str(bet.get("selection") or "").strip()
        taken_odds = bet.get("bookmaker_odds")
        bet_id = bet.get("id")

        if not taken_odds or taken_odds <= 1 or bet_id is None:
            return "ignores"

        extraction = self._extract_closing_market(event, market, selection)
        if not extraction:
            return "ignores"

        closing_sel, market_odds, sel_idx = extraction

        # Cote juste de clôture : 1 / prob no-vig (Shin)
        probs = novig_probs(market_odds)
        if not probs or not (0 < probs[sel_idx] < 1):
            return "ignores"

        fair_closing = 1.0 / probs[sel_idx]

        # CLV% = (cote prise / cote juste de clôture − 1) × 100
        clv_pct = (taken_odds / fair_closing - 1.0) * 100.0

        self.db.update_bet_closing(
            bet_id, round(closing_sel, 3), round(clv_pct, 2)
        )
        return "captures"

    # ─── EXTRACTION DES COTES DE CLÔTURE ─────────────

    def _extract_closing_market(self, event: Dict, market: str,
                                selection: str
                                ) -> Optional[Tuple[float, List[float], int]]:
        """
        Extrait le marché complet de clôture correspondant au
        pari : Pinnacle en priorité, sinon moyenne des bookmakers.

        Returns:
            (cote_clôture_sélection, cotes_du_marché_complet,
             index_de_la_sélection) ou None si non supporté /
            introuvable.
        """

        market_lower = market.lower()
        sel_lower = selection.lower()

        # ── 1X2 → marché h2h (home / draw / away) ──
        if market == "1X2" or market_lower in ("1x2", "match result"):
            index_map = {"1": 0, "x": 1, "2": 2}
            sel_idx = index_map.get(sel_lower)
            if sel_idx is None:
                return None

            labels = [
                event.get("home_team", ""),
                "Draw",
                event.get("away_team", ""),
            ]
            market_odds = self._collect_market_odds(
                event, "h2h", labels, point=None
            )

        # ── Over/Under 2.5 → marché totals (point = 2.5) ──
        elif "2.5" in market:
            if "over" in sel_lower:
                sel_idx = 0
            elif "under" in sel_lower:
                sel_idx = 1
            else:
                return None

            market_odds = self._collect_market_odds(
                event, "totals", ["Over", "Under"], point=2.5
            )

        # ── Autres marchés (BTTS…) : non supportés ──
        else:
            return None

        if not market_odds:
            return None

        return market_odds[sel_idx], market_odds, sel_idx

    def _collect_market_odds(self, event: Dict, market_key: str,
                             labels: List[str],
                             point: Optional[float]
                             ) -> Optional[List[float]]:
        """
        Rassemble les cotes d'un marché complet chez chaque
        bookmaker de l'événement.

        Args:
            event: événement The Odds API
            market_key: "h2h" ou "totals"
            labels: noms des issues dans l'ordre attendu
                (ex: [home, "Draw", away] ou ["Over", "Under"])
            point: ligne du marché totals (2.5) ou None pour h2h

        Returns:
            Cotes du marché complet dans l'ordre de labels —
            celles de Pinnacle si disponibles, sinon la moyenne
            des bookmakers. None si aucun bookmaker complet.
        """

        vectors: Dict[str, List[float]] = {}

        for bookmaker in event.get("bookmakers", []):
            bk_name = str(bookmaker.get("key", "")).lower()

            for mk in bookmaker.get("markets", []):
                if mk.get("key") != market_key:
                    continue

                odds = [0.0] * len(labels)
                for outcome in mk.get("outcomes", []):
                    if point is not None:
                        # Marché totals : ne garder que la bonne ligne
                        o_point = outcome.get("point")
                        if o_point is None or abs(float(o_point) - point) > 1e-9:
                            continue

                    name = str(outcome.get("name", ""))
                    try:
                        price = float(outcome.get("price", 0))
                    except (TypeError, ValueError):
                        continue

                    for i, label in enumerate(labels):
                        if name == label:
                            odds[i] = price
                            break

                # Marché complet uniquement (toutes issues cotées)
                if all(o > 1 for o in odds):
                    vectors[bk_name] = odds

        if not vectors:
            return None

        # Pinnacle en priorité (cote de clôture de référence)
        if "pinnacle" in vectors:
            return vectors["pinnacle"]

        # Sinon : moyenne des bookmakers, issue par issue
        n = len(vectors)
        return [
            sum(v[i] for v in vectors.values()) / n
            for i in range(len(labels))
        ]

    # ─── RECHERCHE D'ÉVÉNEMENT ───────────────────────

    def _find_event(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Retrouve l'événement The Odds API correspondant à un match
        en essayant chaque ligue de LEAGUE_KEYS (une seule requête
        par ligue et par appel grâce au cache mémoire).

        Match validé si token_sort_ratio ≥ FUZZY_THRESHOLD sur les
        DEUX noms d'équipes.
        """

        if not home_team or not away_team:
            return None

        for league, sport_key in self.LEAGUE_KEYS.items():
            events = self._fetch_league_events(sport_key)
            if not events:
                continue

            for event in events:
                ev_home = str(event.get("home_team") or "")
                ev_away = str(event.get("away_team") or "")

                score_home = fuzz.token_sort_ratio(
                    home_team.lower(), ev_home.lower()
                )
                score_away = fuzz.token_sort_ratio(
                    away_team.lower(), ev_away.lower()
                )

                if (score_home >= self.FUZZY_THRESHOLD
                        and score_away >= self.FUZZY_THRESHOLD):
                    return event

        return None

    def _fetch_league_events(self, sport_key: str) -> List[Dict]:
        """
        Récupère (avec cache mémoire) les événements et cotes d'une
        ligue : GET /v4/sports/{key}/odds. Une ligue déjà interrogée
        — même en échec — n'est jamais re-demandée par la même
        instance : économie stricte du quota.
        """

        if sport_key in self._events_cache:
            return self._events_cache[sport_key] or []

        events: List[Dict] = []

        try:
            response = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "eu",
                    "markets": "h2h,totals",
                    "oddsFormat": "decimal",
                },
                timeout=30,
            )

            # Suivre le quota restant
            remaining = response.headers.get("x-requests-remaining")
            if remaining is not None:
                self._remaining_requests = remaining

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    events = data
            elif response.status_code == 401:
                print("   ❌ Clé The Odds API invalide")
            elif response.status_code == 429:
                print("   ⏳ Quota The Odds API épuisé")
            else:
                print(f"   ❌ Erreur The Odds API "
                      f"({sport_key}): {response.status_code}")

        except Exception as e:
            print(f"   ❌ Erreur connexion The Odds API ({sport_key}): {e}")

        self._events_cache[sport_key] = events
        return events
