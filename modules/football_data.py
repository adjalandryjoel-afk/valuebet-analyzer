"""
═══════════════════════════════════════════════════════
 MODULE FOOTBALL-DATA — données réelles des championnats
 européens (football-data.co.uk)
═══════════════════════════════════════════════════════

Source gratuite, sans clé et sans quota, mise à jour en
continu, qui couvre 16 championnats européens depuis les
années 1990 avec, pour CHAQUE match :
  score final, score mi-temps, tirs, TIRS CADRÉS, corners,
  cartons, et les cotes de clôture Pinnacle.

C'est la colonne vertébrale factuelle de l'app :
  • paramètres réels de chaque ligue (buts, avantage du
    terrain, part de la 1ère mi-temps) — mesurés, plus estimés
  • forme récente d'une équipe — résultats réels
  • confrontations directes (H2H) — vraies rencontres
  • profil de tirs cadrés par équipe — le modèle cesse de
    deviner les tirs à partir des buts

Remplace API-Football (abonnement RapidAPI expiré, et dont
le plan gratuit ne donnait de toute façon pas la saison en
cours). Aucune clé, donc rien qui puisse expirer.

Contrairement à API-Football, cette source n'impose ni clé, ni
quota, ni compte : le cloud Streamlit télécharge donc directement
(CSV mis en cache 12 h sur disque). Si le réseau échoue, les
méthodes renvoient None et les agents affichent simplement
« indisponible » — jamais de données inventées.

Les paramètres de ligue, eux, sont figés dans data/league_params.json
(committé) : ils sont disponibles même sans réseau.
"""

import io
import json
import os
import time
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from config import Paths

BASE_URL = "https://www.football-data.co.uk/mmz4281"
# Second jeu de données football-data (« new ») : un fichier par
# pays, couvrant beaucoup plus de championnats (Scandinavie, Europe
# de l'Est, Amériques...) avec scores et cotes — mais SANS mi-temps
# ni tirs cadrés. Suffisant pour la forme et les confrontations.
EXTRA_URL = "https://www.football-data.co.uk/new"
CACHE_DIR = os.path.join(Paths.DATA_DIR, "football_data")

# Colonnes exploitées (les autres sont ignorées)
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HST", "AST"]

# Appariement des noms : un match approximatif accepté à tort fait
# analyser LA MAUVAISE ÉQUIPE en silence (« Paris Saint-Germain » →
# « Paris FC »). Règle : très bon score, OU bon score nettement
# détaché du suivant. Sinon on refuse — pas de données vaut mieux
# que de fausses données.
FUZZY_SUR = 85      # score au-delà duquel on accepte sans réserve
FUZZY_MIN = 60      # score minimum, s'il devance nettement le suivant
FUZZY_MARGE = 10    # avance requise sur le 2ème candidat

# Noms verbeux ou pièges, par ligue : football-data emploie des noms
# courts, et plusieurs clubs d'une même ligue partagent un préfixe
# (Paris SG/Paris FC, Ath Madrid/Ath Bilbao, Club/Cercle Brugge,
# Sporting Lisbonne/Braga, les deux Borussia...). Ces cas ne peuvent
# pas être tranchés par similarité : ils sont déclarés.
ALIASES: Dict[str, Dict[str, str]] = {
    "ligue1_fr": {
        "paris saint germain": "Paris SG", "psg": "Paris SG",
        "paris saint germain fc": "Paris SG",
        "paris fc": "Paris FC",
        "olympique de marseille": "Marseille", "om": "Marseille",
        "olympique lyonnais": "Lyon", "ol": "Lyon",
        "as saint etienne": "St Etienne", "saint etienne": "St Etienne",
        "asse": "St Etienne",
        "stade rennais": "Rennes", "stade de reims": "Reims",
        "losc lille": "Lille", "losc": "Lille", "rc lens": "Lens",
        "ogc nice": "Nice", "as monaco": "Monaco", "fc nantes": "Nantes",
        "stade brestois": "Brest", "montpellier hsc": "Montpellier",
        "toulouse fc": "Toulouse", "rc strasbourg": "Strasbourg",
        "fc metz": "Metz", "aj auxerre": "Auxerre", "angers sco": "Angers",
        "fc lorient": "Lorient", "le havre ac": "Le Havre",
    },
    "la_liga": {
        "atletico madrid": "Ath Madrid", "atletico de madrid": "Ath Madrid",
        "atletico": "Ath Madrid",
        "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao",
        "athletic": "Ath Bilbao",
        "real sociedad": "Sociedad", "real betis": "Betis",
        "fc barcelone": "Barcelona", "barcelone": "Barcelona",
        "barca": "Barcelona", "fc barcelona": "Barcelona",
        "seville": "Sevilla", "fc seville": "Sevilla",
        "celta vigo": "Celta", "rayo vallecano": "Vallecano",
        "espanyol": "Espanol", "espanyol barcelone": "Espanol",
        "deportivo alaves": "Alaves", "real valladolid": "Valladolid",
        "ud las palmas": "Las Palmas", "real oviedo": "Oviedo",
    },
    "premier_league": {
        "manchester united": "Man United", "man utd": "Man United",
        "manchester city": "Man City",
        "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
        "nottingham forest": "Nott'm Forest",
        "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
        "newcastle united": "Newcastle", "west ham united": "West Ham",
        "brighton and hove albion": "Brighton",
        "brighton hove albion": "Brighton",
        "leeds united": "Leeds", "leicester city": "Leicester",
        "afc bournemouth": "Bournemouth", "aston villa": "Aston Villa",
    },
    "serie_a": {
        "inter milan": "Inter", "internazionale": "Inter",
        "inter de milan": "Inter",
        "ac milan": "Milan", "milan ac": "Milan",
        "as roma": "Roma", "ss lazio": "Lazio",
        "ssc naples": "Napoli", "naples": "Napoli",
        "juventus turin": "Juventus", "juve": "Juventus",
        "hellas verone": "Verona", "verone": "Verona",
        "la fiorentina": "Fiorentina", "atalanta bergame": "Atalanta",
        "bologne": "Bologna", "genes": "Genoa", "turin": "Torino",
    },
    "bundesliga": {
        "borussia dortmund": "Dortmund", "bvb": "Dortmund",
        "borussia monchengladbach": "M'gladbach",
        "monchengladbach": "M'gladbach", "gladbach": "M'gladbach",
        "eintracht francfort": "Ein Frankfurt",
        "eintracht frankfurt": "Ein Frankfurt",
        "bayern munich": "Bayern Munich", "bayern": "Bayern Munich",
        "fc cologne": "FC Koln", "cologne": "FC Koln", "koln": "FC Koln",
        "bayer leverkusen": "Leverkusen",
        "rb leipzig": "RB Leipzig", "leipzig": "RB Leipzig",
        "vfb stuttgart": "Stuttgart", "vfl wolfsburg": "Wolfsburg",
        "sc fribourg": "Freiburg", "fribourg": "Freiburg",
        "werder breme": "Werder Bremen", "breme": "Werder Bremen",
        "fc saint pauli": "St Pauli", "saint pauli": "St Pauli",
        "hambourg": "Hamburg", "hambourg sv": "Hamburg",
    },
    "eredivisie": {
        "ajax amsterdam": "Ajax", "psv": "PSV Eindhoven",
        "psv eindhoven": "PSV Eindhoven",
        "az alkmaar": "AZ Alkmaar", "fortuna sittard": "For Sittard",
        "nec nimegue": "Nijmegen", "nimegue": "Nijmegen",
        "sparta rotterdam": "Sparta Rotterdam",
        "pec zwolle": "Zwolle", "rkc waalwijk": "Waalwijk",
        "go ahead eagles": "Go Ahead Eagles",
    },
    "primeira_liga": {
        "sporting cp": "Sp Lisbon", "sporting lisbonne": "Sp Lisbon",
        "sporting portugal": "Sp Lisbon", "sporting": "Sp Lisbon",
        "sporting braga": "Sp Braga", "sc braga": "Sp Braga",
        "braga": "Sp Braga",
        "benfica lisbonne": "Benfica", "sl benfica": "Benfica",
        "fc porto": "Porto", "vitoria guimaraes": "Guimaraes",
    },
    "belgium_pro": {
        "club bruges": "Club Brugge", "club brugge": "Club Brugge",
        "cercle bruges": "Cercle Brugge", "cercle brugge": "Cercle Brugge",
        "rsc anderlecht": "Anderlecht", "royal antwerp": "Antwerp",
        "standard de liege": "Standard", "standard liege": "Standard",
        "union saint gilloise": "St. Gilloise",
        "saint gilloise": "St. Gilloise",
        "saint trond": "St Truiden", "krc genk": "Genk",
        "la gantoise": "Gent", "gand": "Gent",
    },
}


class FootballData:
    """Accès aux données réelles des championnats européens."""

    # clé interne de ligue → division football-data.co.uk
    DIVISIONS = {
        "premier_league": "E0",
        "championship": "E1",
        "scotland_prem": "SC0",
        "bundesliga": "D1",
        "bundesliga2": "D2",
        "serie_a": "I1",
        "serie_b": "I2",
        "la_liga": "SP1",
        "la_liga2": "SP2",
        "ligue1_fr": "F1",
        "ligue2_fr": "F2",
        "eredivisie": "N1",
        "belgium_pro": "B1",
        "primeira_liga": "P1",
        "super_lig": "T1",
        "greece_super": "G1",
    }

    # Championnats du 2e jeu « new » : clé interne → (code pays,
    # nom de division dans le fichier). Un seul fichier par pays,
    # toutes saisons confondues. Scores uniquement (pas de mi-temps
    # ni de tirs cadrés) → forme et H2H, pas de profil de tirs.
    EXTRA_LEAGUES = {
        "swe_allsvenskan":  ("SWE", "Allsvenskan"),
        "nor_eliteserien":  ("NOR", "Eliteserien"),
        "fin_veikkaus":     ("FIN", "Veikkausliiga"),
        "den_superliga":    ("DNK", "Superliga"),
        "rus_premier":      ("RUS", "Premier League"),
        "rou_superliga":    ("ROU", "Superliga"),
        "aut_bundesliga":   ("AUT", "Bundesliga"),
        "pol_ekstraklasa":  ("POL", "Ekstraklasa"),
        # NB : football-data n'a PAS l'Islande — son fichier « ISL »
        # sert en réalité les données irlandaises. Islande = non
        # couverte (mieux vaut « indisponible » qu'une fausse équipe).
        "irl_premier":      ("IRL", "Premier Division"),
        "usa_mls":          ("USA", "MLS"),
        "bra_serie_a":      ("BRA", "Serie A"),
        "arg_liga":         ("ARG", "Liga Profesional"),
        "mex_liga":         ("MEX", "Liga MX"),
    }

    # Nombre de saisons chargées (la plus récente d'abord)
    N_SEASONS = 3
    # Un CSV en cache est rafraîchi passé ce délai (saison en cours)
    CACHE_TTL_SECONDS = 12 * 3600
    # En dessous, la saison est jugée non commencée / inexploitable
    MIN_MATCHS = 40

    def __init__(self, offline: bool = False):
        """
        offline=True : n'utilise que les CSV déjà en cache disque
        (aucune requête réseau). Utile pour les tests.
        """
        self.offline = offline
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._frames: Dict[str, pd.DataFrame] = {}

    # ─── SAISONS ─────────────────────────────────────

    @staticmethod
    def season_codes(n: int = 3) -> List[str]:
        """
        Codes des n dernières saisons, la plus récente d'abord.
        Une saison européenne démarre en juillet : en juillet 2026
        la saison courante est 2026/27 → "2627".
        """
        now = datetime.now()
        debut = now.year if now.month >= 7 else now.year - 1
        return [f"{str(a)[2:]}{str(a + 1)[2:]}"
                for a in range(debut, debut - n, -1)]

    # ─── CHARGEMENT ──────────────────────────────────

    def _csv(self, div: str, season: str) -> Optional[pd.DataFrame]:
        """Un CSV de division/saison, avec cache disque."""

        path = os.path.join(CACHE_DIR, f"{div}_{season}.csv")
        frais = (os.path.exists(path)
                 and time.time() - os.path.getmtime(path)
                 < self.CACHE_TTL_SECONDS)

        if not frais and not self.offline:
            try:
                r = requests.get(f"{BASE_URL}/{season}/{div}.csv",
                                 headers={"User-Agent": "Mozilla/5.0"},
                                 timeout=60)
                if r.status_code == 200 and len(r.content) > 200:
                    with open(path, "wb") as f:
                        f.write(r.content)
                elif not os.path.exists(path):
                    return None  # saison pas encore publiée
            except requests.RequestException:
                if not os.path.exists(path):
                    return None  # réseau KO et rien en cache

        if not os.path.exists(path):
            return None

        try:
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            return None

        manquantes = [c for c in ("HomeTeam", "AwayTeam", "FTHG", "FTAG")
                      if c not in df.columns]
        if manquantes:
            return None

        for c in COLS:
            if c not in df.columns:
                df[c] = None
        df = df[COLS].dropna(subset=["HomeTeam", "AwayTeam",
                                     "FTHG", "FTAG"])
        if df.empty:
            return None

        df["saison"] = season
        return df

    def frame(self, league: str) -> Optional[pd.DataFrame]:
        """
        Tous les matchs des N dernières saisons d'une ligue,
        du plus ancien au plus récent.
        """

        if league in self._frames:
            return self._frames[league]

        if league in self.EXTRA_LEAGUES:
            df = self._extra_frame(league)
            if df is not None:
                self._frames[league] = df
            return df

        div = self.DIVISIONS.get(league)
        if not div:
            return None

        morceaux = [df for s in reversed(self.season_codes(self.N_SEASONS))
                    if (df := self._csv(div, s)) is not None]
        if not morceaux:
            return None

        df = pd.concat(morceaux, ignore_index=True)
        df["date_dt"] = pd.to_datetime(df["Date"], dayfirst=True,
                                       errors="coerce")
        df = df.dropna(subset=["date_dt"]).sort_values("date_dt")
        self._frames[league] = df
        return df

    def _extra_frame(self, league: str) -> Optional[pd.DataFrame]:
        """
        Charge un championnat du 2e jeu « new » (un fichier par pays)
        et le ramène au même format interne que les fichiers
        principaux, pour que forme/H2H/detect_league marchent tels
        quels. Mi-temps et tirs cadrés absents (colonnes vides).
        """

        code, division = self.EXTRA_LEAGUES[league]
        path = os.path.join(CACHE_DIR, f"new_{code}.csv")
        frais = (os.path.exists(path)
                 and time.time() - os.path.getmtime(path)
                 < self.CACHE_TTL_SECONDS)

        if not frais and not self.offline:
            try:
                r = requests.get(f"{EXTRA_URL}/{code}.csv",
                                 headers={"User-Agent": "Mozilla/5.0"},
                                 timeout=60)
                if r.status_code == 200 and len(r.content) > 500:
                    with open(path, "wb") as f:
                        f.write(r.content)
            except requests.RequestException:
                pass

        if not os.path.exists(path):
            return None
        try:
            raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        except Exception:
            return None

        besoin = {"League", "Season", "Date", "Home", "Away", "HG", "AG"}
        if not besoin.issubset(raw.columns):
            return None

        # Division exacte (les noms peuvent avoir des espaces parasites)
        raw["_lg"] = raw["League"].astype(str).str.strip()
        raw = raw[raw["_lg"] == division]
        raw = raw.dropna(subset=["Home", "Away", "HG", "AG"])
        if raw.empty:
            return None

        # N dernières saisons présentes dans le fichier
        saisons = sorted(raw["Season"].dropna().astype(str).unique())
        raw = raw[raw["Season"].astype(str).isin(saisons[-self.N_SEASONS:])]

        df = pd.DataFrame({
            "Date": raw["Date"],
            "HomeTeam": raw["Home"].astype(str).str.strip(),
            "AwayTeam": raw["Away"].astype(str).str.strip(),
            "FTHG": pd.to_numeric(raw["HG"], errors="coerce"),
            "FTAG": pd.to_numeric(raw["AG"], errors="coerce"),
            "FTR": raw.get("Res"),
            "HTHG": None, "HTAG": None, "HST": None, "AST": None,
            "saison": raw["Season"].astype(str),
        })
        df = df.dropna(subset=["FTHG", "FTAG"])
        df["date_dt"] = pd.to_datetime(df["Date"], dayfirst=True,
                                       errors="coerce")
        df = df.dropna(subset=["date_dt"]).sort_values("date_dt")
        return df if not df.empty else None

    # ─── NOMS D'ÉQUIPES ──────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        s = unicodedata.normalize("NFKD", str(name or ""))
        s = "".join(c for c in s if not unicodedata.combining(c))
        return " ".join(s.lower().replace("-", " ").split())

    def teams(self, league: str) -> List[str]:
        df = self.frame(league)
        if df is None:
            return []
        return sorted(set(df.HomeTeam) | set(df.AwayTeam))

    def match_team(self, league: str, name: str,
                   strict: bool = False) -> Optional[str]:
        """
        Nom football-data correspondant à un nom Betclic
        (« Paris Saint-Germain » → « Paris SG »).

        Retourne None si le nom est ambigu : dans une app qui engage
        de l'argent, analyser la mauvaise équipe est bien pire que
        n'afficher aucune donnée.

        strict=True : pour DEVINER la ligue d'un match (detect_league),
        on n'accepte que exact / alias / score ≥ FUZZY_SUR. La règle de
        « netteté » (marge sur le 2e) mesure la distinction DANS une
        ligue, pas l'APPARTENANCE : pour une équipe absente, le meilleur
        résident aléatoire devance presque toujours le 2e → faux positif
        (« Santos » apparié « Nantes »). En mode strict on refuse ce cas.
        """

        equipes = self.teams(league)
        if not equipes:
            return None

        cible = self._normalize(name)
        index = {self._normalize(e): e for e in equipes}

        # 1. Nom exact
        if cible in index:
            return index[cible]

        # 2. Alias déclaré (pièges et noms verbeux)
        alias = ALIASES.get(league, {}).get(cible)
        if alias and alias in equipes:
            return alias

        # 3. Similarité, avec exigence de netteté
        scores = process.extract(cible, list(index.keys()),
                                 scorer=fuzz.token_sort_ratio, limit=2)
        if not scores:
            return None

        meilleur, score = scores[0][0], scores[0][1]
        second = scores[1][1] if len(scores) > 1 else 0

        if score >= FUZZY_SUR:
            return index[meilleur]
        if strict:
            return None  # inférence de ligue : exact / alias / ≥85 only
        if score >= FUZZY_MIN and score - second >= FUZZY_MARGE:
            return index[meilleur]

        return None  # ambigu → on refuse

    # ─── DÉTECTION DE LIGUE ──────────────────────────

    # Termes de la compétition (OCR / saisie) → clé de ligue. C'est
    # le signal le plus fiable : Betclic affiche le championnat sur
    # chaque capture. Recherché en minuscules sans accents.
    COMPETITION_HINTS = {
        "ligue 1": "ligue1_fr", "ligue1": "ligue1_fr",
        "ligue 2": "ligue2_fr",
        "premier league": "premier_league", "angleterre": "premier_league",
        "championship": "championship",
        "liga": "la_liga", "la liga": "la_liga", "espagne": "la_liga",
        "segunda": "la_liga2",
        "serie a": "serie_a", "italie": "serie_a",
        "serie b": "serie_b",
        "bundesliga 2": "bundesliga2", "2 bundesliga": "bundesliga2",
        "bundesliga": "bundesliga", "allemagne": "bundesliga",
        "eredivisie": "eredivisie", "pays bas": "eredivisie",
        "pro league": "belgium_pro", "belgique": "belgium_pro",
        "jupiler": "belgium_pro",
        "primeira": "primeira_liga", "portugal": "primeira_liga",
        "liga portugal": "primeira_liga",
        "super lig": "super_lig", "turquie": "super_lig",
        "super league grece": "greece_super", "grece": "greece_super",
        "premiership": "scotland_prem", "ecosse": "scotland_prem",
        # Championnats supplémentaires (2e jeu de données)
        "allsvenskan": "swe_allsvenskan", "suede": "swe_allsvenskan",
        "eliteserien": "nor_eliteserien", "norvege": "nor_eliteserien",
        "veikkausliiga": "fin_veikkaus", "finlande": "fin_veikkaus",
        "superligaen": "den_superliga", "danemark": "den_superliga",
        "premier liga russie": "rus_premier", "russie": "rus_premier",
        "superliga roumanie": "rou_superliga", "roumanie": "rou_superliga",
        "bundesliga autriche": "aut_bundesliga", "autriche": "aut_bundesliga",
        "ekstraklasa": "pol_ekstraklasa", "pologne": "pol_ekstraklasa",
        "irlande": "irl_premier",
        "mls": "usa_mls",
        "brasileirao": "bra_serie_a", "serie a bresil": "bra_serie_a",
        "serie a brazil": "bra_serie_a", "bresil": "bra_serie_a",
        "liga profesional": "arg_liga", "argentine": "arg_liga",
        "liga mx": "mex_liga", "mexique": "mex_liga",
    }

    def league_from_competition(self, competition: str) -> Optional[str]:
        """Clé de ligue déduite du libellé de compétition (OCR)."""
        if not competition:
            return None
        texte = self._normalize(competition)
        # Priorité aux libellés les plus longs (« ligue 2 » avant
        # « ligue », « bundesliga 2 » avant « bundesliga »)
        for hint in sorted(self.COMPETITION_HINTS, key=len, reverse=True):
            if hint in texte:
                return self.COMPETITION_HINTS[hint]
        return None

    # Grands championnats d'abord : la recherche s'arrête au premier
    # où les DEUX équipes sont trouvées, donc on teste les ligues les
    # plus probables en premier (et on évite de tout télécharger).
    _ORDRE_RECHERCHE = [
        "premier_league", "la_liga", "serie_a", "bundesliga",
        "ligue1_fr", "eredivisie", "primeira_liga", "belgium_pro",
        "super_lig", "championship", "scotland_prem", "greece_super",
        "bundesliga2", "serie_b", "la_liga2", "ligue2_fr",
        # Championnats supplémentaires (2e jeu de données)
        "swe_allsvenskan", "nor_eliteserien", "fin_veikkaus",
        "den_superliga", "rus_premier", "rou_superliga",
        "aut_bundesliga", "pol_ekstraklasa",
        "irl_premier", "usa_mls", "bra_serie_a", "arg_liga", "mex_liga",
    ]

    def detect_league(self, home: str, away: str,
                      competition: str = "") -> Optional[str]:
        """
        Ligue d'un match, sans dépendre d'une base d'équipes locale.

        1. Libellé de compétition (fiable, gratuit, instantané)
        2. Sinon : la ligue où les DEUX équipes existent réellement
           dans football-data (les listes sont complètes).
        Retourne None si aucune ne contient les deux équipes.
        """

        # 1. Indice de compétition — le signal le PLUS fiable
        #    (Betclic nomme le championnat). Il prime sur le scan.
        cle = self.league_from_competition(competition)
        if cle and (cle in self.DIVISIONS or cle in self.EXTRA_LEAGUES):
            # Confirmé si les deux équipes y figurent ; sinon on garde
            # l'indice quand même (noms écrits différemment).
            if (self.match_team(cle, home) and self.match_team(cle, away)):
                return cle
            indice = cle
        else:
            indice = None

        # 2. Recherche STRICTE : la ligue où les DEUX équipes existent
        #    vraiment (exact/alias/≥85). En mode non strict, la règle de
        #    marge fait apparier une équipe absente au meilleur résident
        #    → mauvaise ligue ET mauvaises équipes en silence.
        for lg in self._ORDRE_RECHERCHE:
            if (self.match_team(lg, home, strict=True)
                    and self.match_team(lg, away, strict=True)):
                return lg

        return indice  # l'indice de compétition, à défaut de confirmation

    # ─── PARAMÈTRES DE LIGUE ─────────────────────────

    def league_params(self, league: str) -> Optional[Dict]:
        """
        Paramètres réels d'une ligue, mesurés sur la saison
        complète la plus récente : buts par match, taux de
        victoire à domicile, part des buts en 1ère mi-temps.
        """

        df = self.frame(league)
        if df is None:
            return None

        # Saison complète la plus récente (assez de matchs)
        for saison in reversed(sorted(df.saison.unique())):
            sdf = df[df.saison == saison]
            if len(sdf) >= self.MIN_MATCHS:
                break
        else:
            return None

        buts = (sdf.FTHG + sdf.FTAG)
        params = {
            "avg_goals": round(float(buts.mean()), 3),
            # Buts moyens de l'équipe À DOMICILE et de l'équipe À
            # L'EXTÉRIEUR : dénominateurs par venue pour normaliser
            # les splits sans réinjecter l'avantage du terrain.
            "avg_goals_home": round(float(sdf.FTHG.mean()), 3),
            "avg_goals_away": round(float(sdf.FTAG.mean()), 3),
            "home_win_rate": round(float((sdf.FTR == "H").mean()), 3),
            "matchs": int(len(sdf)),
            "saison": str(saison),
        }

        mt = sdf.dropna(subset=["HTHG", "HTAG"])
        if len(mt) >= self.MIN_MATCHS and (mt.FTHG + mt.FTAG).sum() > 0:
            part = ((mt.HTHG + mt.HTAG).sum()
                    / (mt.FTHG + mt.FTAG).sum())
            params["first_half_share"] = round(float(part), 3)

        # Tirs cadrés moyens PAR ÉQUIPE et par match : c'est la
        # référence qui permet de normaliser attaque × défense,
        # exactement comme avg_goals le fait pour les buts.
        sot = sdf.dropna(subset=["HST", "AST"])
        if len(sot) >= self.MIN_MATCHS:
            params["avg_sot"] = round(
                float((sot.HST + sot.AST).mean() / 2), 3)
            params["sot_par_but"] = round(
                float((sot.HST + sot.AST).sum()
                      / max((sot.FTHG + sot.FTAG).sum(), 1)), 3)

        return params

    # ─── PROFIL D'UNE ÉQUIPE ─────────────────────────

    def team_profile(self, league: str, team: str,
                     n_matchs: int = 38) -> Optional[Dict]:
        """
        Profil d'une équipe sur ses n derniers matchs : buts et
        TIRS CADRÉS marqués/encaissés, à domicile et à l'extérieur.

        Les tirs cadrés sont la donnée qui manquait au modèle :
        il les déduisait des buts (× 3.1), une approximation qui
        dérivait du réel.
        """

        df = self.frame(league)
        if df is None:
            return None
        nom = self.match_team(league, team)
        if not nom:
            return None

        rows = df[(df.HomeTeam == nom) | (df.AwayTeam == nom)].tail(n_matchs)
        if len(rows) < 5:
            return None

        dom = rows[rows.HomeTeam == nom]
        ext = rows[rows.AwayTeam == nom]

        def moy(serie):
            serie = serie.dropna()
            return round(float(serie.mean()), 3) if len(serie) else None

        profil = {
            "equipe": nom,
            "matchs": int(len(rows)),
            "buts_pour": moy(pd.concat([dom.FTHG, ext.FTAG])),
            "buts_contre": moy(pd.concat([dom.FTAG, ext.FTHG])),
            "buts_pour_dom": moy(dom.FTHG),
            "buts_contre_dom": moy(dom.FTAG),
            "buts_pour_ext": moy(ext.FTAG),
            "buts_contre_ext": moy(ext.FTHG),
            "sot_pour": moy(pd.concat([dom.HST, ext.AST])),
            "sot_contre": moy(pd.concat([dom.AST, ext.HST])),
            "sot_pour_dom": moy(dom.HST),
            "sot_pour_ext": moy(ext.AST),
        }
        return profil

    # ─── FORME RÉCENTE ───────────────────────────────

    def team_form(self, league: str, team: str,
                  n: int = 6) -> Optional[Dict]:
        """
        Forme réelle : n derniers résultats, séquence VNDD…,
        points, buts marqués/encaissés.
        """

        df = self.frame(league)
        if df is None:
            return None
        nom = self.match_team(league, team)
        if not nom:
            return None

        rows = df[(df.HomeTeam == nom) | (df.AwayTeam == nom)].tail(n)
        if rows.empty:
            return None

        details, seq, pts, bp, bc = [], [], 0, 0, 0
        for _, r in rows.iterrows():
            dom = r.HomeTeam == nom
            pour, contre = (int(r.FTHG), int(r.FTAG)) if dom \
                else (int(r.FTAG), int(r.FTHG))
            lettre = "V" if pour > contre else ("N" if pour == contre else "D")
            pts += 3 if lettre == "V" else (1 if lettre == "N" else 0)
            bp += pour
            bc += contre
            seq.append(lettre)
            details.append({
                "date": r["date_dt"].strftime("%d/%m/%Y"),
                "adversaire": r.AwayTeam if dom else r.HomeTeam,
                "venue": "domicile" if dom else "extérieur",
                "score": f"{pour}-{contre}",
                "resultat": lettre,
            })

        return {
            "equipe": nom,
            "sequence": "".join(seq),
            "points": pts,
            "sur": len(seq) * 3,
            "buts_marques": bp,
            "buts_encaisses": bc,
            "matchs": details,
        }

    # ─── CONFRONTATIONS DIRECTES ─────────────────────

    def h2h(self, league: str, home: str, away: str,
            limit: int = 10) -> Optional[Dict]:
        """
        Vraies confrontations directes des N dernières saisons,
        toutes compétitions de cette ligue confondues.
        """

        df = self.frame(league)
        if df is None:
            return None
        h = self.match_team(league, home)
        a = self.match_team(league, away)
        if not h or not a:
            return None

        rows = df[((df.HomeTeam == h) & (df.AwayTeam == a))
                  | ((df.HomeTeam == a) & (df.AwayTeam == h))].tail(limit)
        if rows.empty:
            return {"equipes": [h, a], "total": 0, "matchs": []}

        v_h = v_a = nuls = btts = 0
        buts, marques_h, marques_a = [], [], []
        details = []
        for _, r in rows.iterrows():
            hg, ag = int(r.FTHG), int(r.FTAG)
            buts.append(hg + ag)
            if hg > 0 and ag > 0:
                btts += 1

            # Buts de CHAQUE équipe, quel que soit qui recevait
            if r.HomeTeam == h:
                marques_h.append(hg)
                marques_a.append(ag)
            else:
                marques_h.append(ag)
                marques_a.append(hg)

            if hg == ag:
                nuls += 1
            elif (r.HomeTeam == h) == (hg > ag):
                v_h += 1
            else:
                v_a += 1

            details.append({
                "date": r["date_dt"].strftime("%d/%m/%Y"),
                "domicile": r.HomeTeam,
                "exterieur": r.AwayTeam,
                "score": f"{hg}-{ag}",
            })

        n = len(details)
        return {
            "equipes": [h, a],
            "total": n,
            "victoires_home": v_h,
            "victoires_away": v_a,
            "nuls": nuls,
            "moy_buts": round(sum(buts) / n, 2),
            "buts_home": round(sum(marques_h) / n, 2),
            "buts_away": round(sum(marques_a) / n, 2),
            "btts_rate": round(btts / n, 3),
            "matchs": details,
        }

# ─── SINGLETON ───────────────────────────────────────

_instance: Optional[FootballData] = None


def get_football_data() -> FootballData:
    """Instance partagée (les DataFrames de ligue sont lourds)."""
    global _instance
    if _instance is None:
        _instance = FootballData()
    return _instance
