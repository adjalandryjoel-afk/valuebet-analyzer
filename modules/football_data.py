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

    def match_team(self, league: str, name: str) -> Optional[str]:
        """
        Nom football-data correspondant à un nom Betclic
        (« Paris Saint-Germain » → « Paris SG »).

        Retourne None si le nom est ambigu : dans une app qui engage
        de l'argent, analyser la mauvaise équipe est bien pire que
        n'afficher aucune donnée.
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

        if score >= FUZZY_SUR or (score >= FUZZY_MIN
                                  and score - second >= FUZZY_MARGE):
            return index[meilleur]

        return None  # ambigu → on refuse

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
            "home_win_rate": round(float((sdf.FTR == "H").mean()), 3),
            "matchs": int(len(sdf)),
            "saison": str(saison),
        }

        mt = sdf.dropna(subset=["HTHG", "HTAG"])
        if len(mt) >= self.MIN_MATCHS and (mt.FTHG + mt.FTAG).sum() > 0:
            part = ((mt.HTHG + mt.HTAG).sum()
                    / (mt.FTHG + mt.FTAG).sum())
            params["first_half_share"] = round(float(part), 3)

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
