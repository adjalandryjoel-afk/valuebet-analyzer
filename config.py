"""
═══════════════════════════════════════════════════════
 CONFIGURATION GLOBALE — Value Bet Analyzer
 Betclic Côte d'Ivoire
═══════════════════════════════════════════════════════

Toutes les constantes et paramètres du logiciel sont
centralisés ici. Les clés API sont lues depuis le .env.
"""

import json
import os
import sys
from dotenv import load_dotenv

# Console Windows : forcer l'UTF-8 pour que les emojis des rapports
# ne fassent pas planter les print() (cp1252 par défaut)
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Charger le .env situé à côté de ce fichier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


# ══════════════════════════════════════════════════════
#  CLÉS API
# ══════════════════════════════════════════════════════

class APIKeys:
    """Clés API chargées depuis le fichier .env."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
    FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")


# ══════════════════════════════════════════════════════
#  CHEMINS
# ══════════════════════════════════════════════════════

class Paths:
    """Chemins des dossiers et fichiers de données."""

    ROOT_DIR = BASE_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    TEAMS_DATABASE = os.path.join(DATA_DIR, "teams_database.json")
    HISTORICAL_DATA = os.path.join(DATA_DIR, "historical_data.json")
    ELO_RATINGS = os.path.join(DATA_DIR, "elo_ratings.json")
    RESULTS_LOG = os.path.join(DATA_DIR, "results_log.json")


# ══════════════════════════════════════════════════════
#  MODÈLE DE POISSON
# ══════════════════════════════════════════════════════

class PoissonConfig:
    """Paramètres du modèle de Poisson."""

    # Nombre max de buts considérés dans la matrice des scores
    MAX_GOALS = 7

    # Avantage domicile (multiplicateur sur le lambda domicile)
    # 1.10 = +10% de buts attendus à domicile
    HOME_ADVANTAGE = 1.10

    # Moyenne de buts par match (défaut si ligue inconnue)
    DEFAULT_LEAGUE_AVG_GOALS = 2.60

    # Nombre de matchs récents pris en compte pour la forme
    RECENT_MATCHES_COUNT = 5

    # Poids de la forme récente vs stats de la saison (0..1)
    RECENT_FORM_WEIGHT = 0.35

    # Poids du marché (cotes) vs stats dans l'estimation des lambdas.
    # Calibré par backtest sur 5052 matchs (football-data.co.uk,
    # 2021-2024) : le log-loss s'améliore de façon monotone jusqu'à
    # 0.9 — le marché est bien plus informatif que les stats maison.
    MARKET_WEIGHT = 0.90

    # Bornes de sécurité sur les lambdas
    MIN_LAMBDA = 0.30
    MAX_LAMBDA = 4.00

    # Part des buts marqués en 1ère mi-temps (défaut si la ligue
    # n'a pas sa propre valeur dans SUPPORTED_LEAGUES)
    FIRST_HALF_SHARE = 0.45

    # Tirs cadrés attendus par but attendu (littérature : ~30-32%
    # des tirs cadrés finissent au fond → ratio central 3.1)
    SOT_PER_GOAL = 3.1

    # Part du xG dans l'estimation de force d'une équipe (le reste =
    # buts réels). Le xG prédit mieux l'avenir que les buts (moins
    # bruité), mais les buts capturent la finition : ~2/3 xG est le
    # compromis retenu par la plupart des modèles xG (ex. SPI).
    XG_BLEND = 0.65

    # Correction Dixon-Coles (1997) : dépendance des scores faibles.
    # rho négatif => gonfle 0-0 et 1-1, dégonfle 1-0 et 0-1.
    # Valeur typique estimée sur les grands championnats : -0.05 à -0.15.
    DIXON_COLES_RHO = -0.10

    # Décote temporelle des matchs passés (Dixon-Coles time decay) :
    # poids = exp(-XI_PER_DAY × ancienneté_en_jours). 0.002/j ≈
    # demi-vie de ~1 an — les vieilles saisons pèsent peu.
    TIME_DECAY_XI = 0.002


# ══════════════════════════════════════════════════════
#  SYSTÈME ELO
# ══════════════════════════════════════════════════════

class EloConfig:
    """Paramètres du système de rating Elo."""

    INITIAL_RATING = 1500
    K_FACTOR = 32

    # Avantage domicile exprimé en points Elo
    HOME_ADVANTAGE_ELO = 60

    # Paramètre du modèle de nul (Davidson) :
    # plus c'est haut, plus la prob de nul de base est élevée
    DRAW_BASE_PROB = 0.26

    # Poids donné à l'estimation par les cotes quand on
    # met à jour un rating existant (0..1)
    ODDS_ESTIMATE_WEIGHT = 0.50


# ══════════════════════════════════════════════════════
#  DÉTECTION DES VALUE BETS
# ══════════════════════════════════════════════════════

class ValueBetConfig:
    """Seuils de détection des value bets."""

    # Value minimum pour signaler un pari (0.05 = +5% d'edge)
    MIN_VALUE_THRESHOLD = 0.05

    # Score de confiance minimum (0-100)
    MIN_CONFIDENCE_SCORE = 55

    # Fourchette de cotes jouables. Plafond abaissé 6.00 → 4.50 :
    # le backtest 2025/26 (grands championnats, 8500 matchs) montre
    # que les value bets sur cotes > 4.5 sont un piège — biais
    # favori-outsider + le modèle qui surestime les outsiders. En
    # coupant cette tranche : ROI passe de -16.7% à -2.4% et CLV
    # (l'indicateur avancé) de -7.0% à -3.6%. Réversible.
    MIN_ODDS = 1.30
    MAX_ODDS = 4.50

    # Seuil de value progressif selon la cote (biais favori-outsider) :
    # multiplicateur appliqué au seuil de base MIN_VALUE_THRESHOLD.
    # (cote_max_exclusive, multiplicateur)
    VALUE_THRESHOLD_MULTIPLIERS = [
        (2.50, 1.0),   # 5% de base sous 2.50
        (4.00, 1.6),   # ~8% entre 2.50 et 4.00
        (99.0, 2.4),   # ~12% au-delà de 4.00
    ]

    # Probabilité modèle minimum (éviter les paris trop improbables)
    MIN_MODEL_PROBABILITY = 0.15

    # Garde-fous anti-erreurs de lecture :
    # au-delà de cette value, c'est quasi certainement une erreur de
    # données (cote mal lue), pas une vraie opportunité
    MAX_PLAUSIBLE_VALUE = 0.40

    # Marge 1X2 au-delà de laquelle les cotes sont jugées incohérentes
    # (un vrai 1X2 a une marge de 2 à 12%) — probable erreur d'extraction
    MAX_SANE_MARGIN = 15.0

    # Poids des modèles dans le blend 1X2
    POISSON_WEIGHT = 0.60
    ELO_WEIGHT = 0.40

    # Étoiles de rating selon la value
    RATING_THRESHOLDS = [
        (0.20, "⭐⭐⭐⭐⭐"),
        (0.15, "⭐⭐⭐⭐"),
        (0.10, "⭐⭐⭐"),
        (0.07, "⭐⭐"),
        (0.05, "⭐"),
    ]


# ══════════════════════════════════════════════════════
#  CRITÈRE DE KELLY (GESTION DE MISE)
# ══════════════════════════════════════════════════════

class KellyConfig:
    """Paramètres de la gestion de bankroll."""

    # Bankroll par défaut en FCFA
    DEFAULT_BANKROLL = 100_000

    # Fraction de Kelly appliquée (0.25 = quart de Kelly, prudent)
    KELLY_FRACTION = 0.25

    # Mise maximum par pari (% du bankroll) — abaissé à 2% tant que
    # le CLV/ROI n'est pas prouvé positif sur 100+ paris (recherche :
    # les drawdowns du quart de Kelly à 5% sont trop violents)
    MAX_STAKE_PERCENTAGE = 2.0

    # Mise minimum en FCFA (en dessous, on ne joue pas)
    MIN_STAKE_AMOUNT = 500

    # Exposition totale maximum sur une session (% du bankroll)
    MAX_TOTAL_EXPOSURE = 20.0

    # Arrondi des mises (FCFA)
    STAKE_ROUNDING = 100


# ══════════════════════════════════════════════════════
#  OCR / EXTRACTION DES CAPTURES
# ══════════════════════════════════════════════════════

class OCRConfig:
    """Paramètres de l'extraction OCR des captures Betclic."""

    SCREENSHOTS_DIR = Paths.SCREENSHOTS_DIR

    # Extensions d'images acceptées
    ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

    # Modèle vision OpenAI utilisé pour l'extraction
    VISION_MODEL = "gpt-4o-mini"

    # Nombre max de captures traitées en une fois
    MAX_IMAGES = 30


# ══════════════════════════════════════════════════════
#  LIGUES SUPPORTÉES
# ══════════════════════════════════════════════════════

# clé interne → infos de la ligue
#
# Les valeurs ci-dessous ne sont que des DÉFAUTS de secours : les
# vraies sont mesurées sur la dernière saison complète et chargées
# depuis data/league_params.json (voir plus bas). Régénérer avec
# scripts/refresh_league_params.py après chaque saison.
SUPPORTED_LEAGUES = {
    # ── Grands championnats ──
    "premier_league": {
        "name": "Premier League", "country": "Angleterre",
        "avg_goals": 2.75, "home_win_rate": 0.43,
        "first_half_share": 0.43,
    },
    "la_liga": {
        "name": "La Liga", "country": "Espagne",
        "avg_goals": 2.70, "home_win_rate": 0.49,
        "first_half_share": 0.43,
    },
    "serie_a": {
        "name": "Serie A", "country": "Italie",
        "avg_goals": 2.43, "home_win_rate": 0.39,
        "first_half_share": 0.43,
    },
    "bundesliga": {
        "name": "Bundesliga", "country": "Allemagne",
        "avg_goals": 3.24, "home_win_rate": 0.44,
        "first_half_share": 0.45,
    },
    "ligue1_fr": {
        "name": "Ligue 1", "country": "France",
        "avg_goals": 2.82, "home_win_rate": 0.46,
        "first_half_share": 0.43,
    },
    # ── Autres championnats européens ──
    "eredivisie": {
        "name": "Eredivisie", "country": "Pays-Bas",
        "avg_goals": 3.18, "home_win_rate": 0.44,
        "first_half_share": 0.45,
    },
    "primeira_liga": {
        "name": "Primeira Liga", "country": "Portugal",
        "avg_goals": 2.68, "home_win_rate": 0.41,
        "first_half_share": 0.45,
    },
    "belgium_pro": {
        "name": "Pro League", "country": "Belgique",
        "avg_goals": 2.68, "home_win_rate": 0.41,
        "first_half_share": 0.45,
    },
    "super_lig": {
        "name": "Süper Lig", "country": "Turquie",
        "avg_goals": 2.65, "home_win_rate": 0.42,
        "first_half_share": 0.43,
    },
    "greece_super": {
        "name": "Super League", "country": "Grèce",
        "avg_goals": 2.57, "home_win_rate": 0.41,
        "first_half_share": 0.47,
    },
    "scotland_prem": {
        "name": "Premiership", "country": "Écosse",
        "avg_goals": 2.78, "home_win_rate": 0.45,
        "first_half_share": 0.45,
    },
    # ── Deuxièmes divisions (lignes plus molles = plus de value) ──
    "championship": {
        "name": "Championship", "country": "Angleterre",
        "avg_goals": 2.61, "home_win_rate": 0.42,
        "first_half_share": 0.46,
    },
    "bundesliga2": {
        "name": "2. Bundesliga", "country": "Allemagne",
        "avg_goals": 2.93, "home_win_rate": 0.46,
        "first_half_share": 0.45,
    },
    "serie_b": {
        "name": "Serie B", "country": "Italie",
        "avg_goals": 2.56, "home_win_rate": 0.45,
        "first_half_share": 0.45,
    },
    "la_liga2": {
        "name": "La Liga 2", "country": "Espagne",
        "avg_goals": 2.63, "home_win_rate": 0.45,
        "first_half_share": 0.45,
    },
    "ligue2_fr": {
        "name": "Ligue 2", "country": "France",
        "avg_goals": 2.54, "home_win_rate": 0.38,
        "first_half_share": 0.47,
    },
    # ── Championnats supplémentaires (2e jeu football-data :
    #    forme + H2H, sans mi-temps ni tirs cadrés) ──
    "swe_allsvenskan": {
        "name": "Allsvenskan", "country": "Suède",
        "avg_goals": 2.95, "home_win_rate": 0.44,
        "first_half_share": 0.45,
    },
    "nor_eliteserien": {
        "name": "Eliteserien", "country": "Norvège",
        "avg_goals": 3.05, "home_win_rate": 0.46,
        "first_half_share": 0.45,
    },
    "fin_veikkaus": {
        "name": "Veikkausliiga", "country": "Finlande",
        "avg_goals": 2.75, "home_win_rate": 0.44,
        "first_half_share": 0.45,
    },
    "den_superliga": {
        "name": "Superliga", "country": "Danemark",
        "avg_goals": 2.90, "home_win_rate": 0.43,
        "first_half_share": 0.45,
    },
    "rus_premier": {
        "name": "Premier Liga", "country": "Russie",
        "avg_goals": 2.55, "home_win_rate": 0.45,
        "first_half_share": 0.45,
    },
    "rou_superliga": {
        "name": "Superliga", "country": "Roumanie",
        "avg_goals": 2.55, "home_win_rate": 0.43,
        "first_half_share": 0.45,
    },
    "aut_bundesliga": {
        "name": "Bundesliga", "country": "Autriche",
        "avg_goals": 3.00, "home_win_rate": 0.45,
        "first_half_share": 0.45,
    },
    "pol_ekstraklasa": {
        "name": "Ekstraklasa", "country": "Pologne",
        "avg_goals": 2.60, "home_win_rate": 0.44,
        "first_half_share": 0.45,
    },
    "irl_premier": {
        "name": "Premier Division", "country": "Irlande",
        "avg_goals": 2.55, "home_win_rate": 0.42,
        "first_half_share": 0.45,
    },
    "usa_mls": {
        "name": "MLS", "country": "États-Unis",
        "avg_goals": 3.05, "home_win_rate": 0.48,
        "first_half_share": 0.45,
    },
    "bra_serie_a": {
        "name": "Brasileirão", "country": "Brésil",
        "avg_goals": 2.45, "home_win_rate": 0.48,
        "first_half_share": 0.45,
    },
    "arg_liga": {
        "name": "Liga Profesional", "country": "Argentine",
        "avg_goals": 2.35, "home_win_rate": 0.42,
        "first_half_share": 0.45,
    },
    "mex_liga": {
        "name": "Liga MX", "country": "Mexique",
        "avg_goals": 2.75, "home_win_rate": 0.47,
        "first_half_share": 0.45,
    },
    # Cotes seules (pas de forme/H2H : 2e division ou sélections)
    "swe_superettan": {
        "name": "Superettan", "country": "Suède",
        "avg_goals": 2.80, "home_win_rate": 0.44,
        "first_half_share": 0.45,
    },
    "world_cup": {
        "name": "Coupe du Monde", "country": "International",
        "avg_goals": 2.65, "home_win_rate": 0.40,
        "first_half_share": 0.45,
    },

    # ── Coupes d'Europe et local (pas de CSV football-data) ──
    "champions_league": {
        "name": "Champions League", "country": "Europe",
        "avg_goals": 2.90, "home_win_rate": 0.45,
        "first_half_share": 0.46,
    },
    "europa_league": {
        "name": "Europa League", "country": "Europe",
        "avg_goals": 2.80, "home_win_rate": 0.44,
        "first_half_share": 0.46,
    },
    "ligue1_ci": {
        "name": "Ligue 1 Côte d'Ivoire", "country": "Côte d'Ivoire",
        "avg_goals": 2.20, "home_win_rate": 0.48,
        "first_half_share": 0.45,
    },
}


# ── Paramètres MESURÉS (écrasent les défauts ci-dessus) ──
# data/league_params.json est produit par scripts/refresh_league_params.py
# à partir des résultats réels de football-data.co.uk, puis committé :
# le cloud en profite sans aucune requête réseau.
def _charger_params_mesures():
    chemin = os.path.join(Paths.DATA_DIR, "league_params.json")
    try:
        with open(chemin, encoding="utf-8") as f:
            mesures = json.load(f).get("ligues", {})
    except (OSError, ValueError):
        return

    for cle, params in mesures.items():
        if cle not in SUPPORTED_LEAGUES:
            continue
        for champ in ("avg_goals", "home_win_rate", "first_half_share",
                      "avg_sot", "sot_par_but"):
            if isinstance(params.get(champ), (int, float)):
                SUPPORTED_LEAGUES[cle][champ] = params[champ]
        SUPPORTED_LEAGUES[cle]["mesure"] = {
            "saison": params.get("saison"),
            "matchs": params.get("matchs"),
        }


_charger_params_mesures()
