"""
═══════════════════════════════════════════════════════
 CONFIGURATION GLOBALE — Value Bet Analyzer
 Betclic Côte d'Ivoire
═══════════════════════════════════════════════════════

Toutes les constantes et paramètres du logiciel sont
centralisés ici. Les clés API sont lues depuis le .env.
"""

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

    # Poids du marché (cotes) vs stats dans l'estimation des lambdas
    # Le marché est très efficient : il reste la meilleure ancre.
    MARKET_WEIGHT = 0.60

    # Bornes de sécurité sur les lambdas
    MIN_LAMBDA = 0.30
    MAX_LAMBDA = 4.00


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

    # Fourchette de cotes jouables
    MIN_ODDS = 1.30
    MAX_ODDS = 7.00

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

    # Mise maximum par pari (% du bankroll)
    MAX_STAKE_PERCENTAGE = 5.0

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
SUPPORTED_LEAGUES = {
    "premier_league": {
        "name": "Premier League",
        "country": "Angleterre",
        "avg_goals": 2.85,
        "home_win_rate": 0.44,
    },
    "la_liga": {
        "name": "La Liga",
        "country": "Espagne",
        "avg_goals": 2.55,
        "home_win_rate": 0.46,
    },
    "serie_a": {
        "name": "Serie A",
        "country": "Italie",
        "avg_goals": 2.65,
        "home_win_rate": 0.42,
    },
    "bundesliga": {
        "name": "Bundesliga",
        "country": "Allemagne",
        "avg_goals": 3.10,
        "home_win_rate": 0.44,
    },
    "ligue1_fr": {
        "name": "Ligue 1",
        "country": "France",
        "avg_goals": 2.60,
        "home_win_rate": 0.45,
    },
    "ligue1_ci": {
        "name": "Ligue 1 Côte d'Ivoire",
        "country": "Côte d'Ivoire",
        "avg_goals": 2.20,
        "home_win_rate": 0.48,
    },
    "champions_league": {
        "name": "Champions League",
        "country": "Europe",
        "avg_goals": 2.90,
        "home_win_rate": 0.45,
    },
    "europa_league": {
        "name": "Europa League",
        "country": "Europe",
        "avg_goals": 2.80,
        "home_win_rate": 0.44,
    },
}
