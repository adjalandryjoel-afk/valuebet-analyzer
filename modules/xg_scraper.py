"""
═══════════════════════════════════════════════════════
 MODULE XG SCRAPER — Collecte des Expected Goals (xG)
 depuis FBref (StatsBomb) et Understat
═══════════════════════════════════════════════════════

Sources :
  • FBref.com → xG fournis par StatsBomb (données les plus fiables)
  • Understat.com → xG par match et par joueur (top 5 ligues)

Les xG sont le paramètre #1 pour la prédiction football moderne.
"""

import re
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import cloudscraper

from config import PoissonConfig


# ══════════════════════════════════════════════════════
#  STRUCTURES DE DONNÉES xG
# ══════════════════════════════════════════════════════

@dataclass
class MatchXG:
    """Données xG d'un match individuel."""

    date: str
    home_team: str
    away_team: str
    home_goals: int = 0
    away_goals: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_xgot: float = 0.0     # xG On Target
    away_xgot: float = 0.0
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_deep_completions: int = 0  # Passes complétées dans la surface
    away_deep_completions: int = 0
    league: str = ""
    source: str = ""


@dataclass
class TeamXGProfile:
    """Profil xG complet d'une équipe sur la saison."""

    team_name: str
    league: str
    season: str

    # xG cumulés
    total_xg_for: float = 0.0
    total_xg_against: float = 0.0
    total_goals_for: int = 0
    total_goals_against: int = 0

    # Moyennes par match
    avg_xg_for: float = 0.0
    avg_xg_against: float = 0.0

    # Différentiel xG
    xg_difference: float = 0.0

    # Sur/sous-performance
    goals_minus_xg: float = 0.0   # Positif = surperformance
    goals_against_minus_xga: float = 0.0

    # xG récents (5 derniers matchs)
    recent_xg_for: List[float] = field(default_factory=list)
    recent_xg_against: List[float] = field(default_factory=list)
    recent_avg_xg_for: float = 0.0
    recent_avg_xg_against: float = 0.0

    # Domicile / Extérieur
    home_avg_xg_for: float = 0.0
    home_avg_xg_against: float = 0.0
    away_avg_xg_for: float = 0.0
    away_avg_xg_against: float = 0.0

    # Données brutes par match
    matches: List[MatchXG] = field(default_factory=list)

    matches_played: int = 0
    data_available: bool = False
    last_updated: str = ""


# ══════════════════════════════════════════════════════
#  SCRAPER FBREF (StatsBomb xG)
# ══════════════════════════════════════════════════════

class FBrefScraper:
    """
    Scrape les données xG depuis FBref.com.

    FBref utilise les données StatsBomb, qui sont considérées
    comme la source xG la plus fiable disponible publiquement.

    Ligues couvertes : Top 5 européennes + quelques autres
    """

    BASE_URL = "https://fbref.com"

    # IDs des ligues sur FBref
    LEAGUE_URLS = {
        "premier_league": "/en/comps/9/Premier-League-Stats",
        "la_liga": "/en/comps/12/La-Liga-Stats",
        "serie_a": "/en/comps/11/Serie-A-Stats",
        "bundesliga": "/en/comps/20/Bundesliga-Stats",
        "ligue1_fr": "/en/comps/13/Ligue-1-Stats",
        "champions_league": "/en/comps/8/Champions-League-Stats",
    }

    # URLs pour les stats de tir (avec xG)
    SHOOTING_URLS = {
        "premier_league": "/en/comps/9/shooting/Premier-League-Stats",
        "la_liga": "/en/comps/12/shooting/La-Liga-Stats",
        "serie_a": "/en/comps/11/shooting/Serie-A-Stats",
        "bundesliga": "/en/comps/20/shooting/Bundesliga-Stats",
        "ligue1_fr": "/en/comps/13/shooting/Ligue-1-Stats",
    }

    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        self.scraper.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        })
        self._request_count = 0
        self.cache = {}

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Récupère et parse une page web avec gestion du rate limit."""

        if url in self.cache:
            return self.cache[url]

        self._request_count += 1

        # Rate limiting : FBref limite à ~20 requêtes/minute
        if self._request_count > 1:
            time.sleep(4)

        try:
            full_url = f"{self.BASE_URL}{url}" if url.startswith("/") else url
            print(f"      🌐 Fetching: {full_url[:80]}...")

            response = self.scraper.get(full_url, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                self.cache[url] = soup
                return soup
            elif response.status_code == 429:
                print(f"      ⏳ Rate limited. Attente 60s...")
                time.sleep(60)
                return self._fetch_page(url)
            else:
                print(f"      ❌ HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"      ❌ Erreur: {e}")
            return None

    # ─── xG PAR ÉQUIPE (TABLEAU DE LA LIGUE) ────────

    def get_league_xg_table(self, league: str) -> Dict[str, TeamXGProfile]:
        """
        Récupère le tableau xG complet d'une ligue.

        Retourne un dict {nom_équipe: TeamXGProfile}
        """

        url = self.SHOOTING_URLS.get(league)
        if not url:
            print(f"      ⚠️ Ligue '{league}' non supportée par FBref")
            return {}

        soup = self._fetch_page(url)
        if not soup:
            return {}

        profiles = {}

        # Trouver le tableau des stats de tir par équipe
        # FBref utilise des commentaires HTML pour les tableaux
        comments = soup.find_all(string=lambda t: isinstance(t, str) and 'stats_shooting' in t)

        table = None

        # Chercher dans le HTML principal d'abord
        table = soup.find('table', {'id': 'stats_shooting'})

        # Si pas trouvé, chercher dans les commentaires
        if not table:
            for comment in soup.find_all(string=lambda t: t and 'stats_shooting' in str(t)):
                comment_soup = BeautifulSoup(str(comment), 'html.parser')
                table = comment_soup.find('table', {'id': 'stats_shooting'})
                if table:
                    break

        if not table:
            # Essayer de trouver n'importe quel tableau avec des colonnes xG
            tables = soup.find_all('table')
            for t in tables:
                headers = t.find_all('th')
                header_texts = [h.get_text() for h in headers]
                if 'xG' in header_texts or 'xg' in [h.lower() for h in header_texts]:
                    table = t
                    break

        if not table:
            print(f"      ⚠️ Tableau xG non trouvé pour {league}")
            return {}

        # Parser le tableau
        tbody = table.find('tbody')
        if not tbody:
            return {}

        rows = tbody.find_all('tr')

        for row in rows:
            if row.get('class') and 'thead' in ' '.join(row.get('class', [])):
                continue

            cells = row.find_all(['td', 'th'])
            if len(cells) < 5:
                continue

            try:
                # Extraire le nom de l'équipe
                team_cell = row.find('td', {'data-stat': 'team'})
                if not team_cell:
                    team_cell = row.find('th', {'data-stat': 'team'})

                if not team_cell:
                    continue

                team_link = team_cell.find('a')
                team_name = team_link.get_text().strip() if team_link else team_cell.get_text().strip()

                # Extraire les stats
                def get_stat(stat_name, default=0):
                    cell = row.find('td', {'data-stat': stat_name})
                    if cell:
                        text = cell.get_text().strip()
                        try:
                            return float(text) if text else default
                        except ValueError:
                            return default
                    return default

                matches_played = int(get_stat('minutes_90s', 0))
                if matches_played == 0:
                    mp_cell = row.find('td', {'data-stat': 'games'})
                    if mp_cell:
                        try:
                            matches_played = int(mp_cell.get_text().strip())
                        except (ValueError, AttributeError):
                            matches_played = 1

                goals = int(get_stat('goals', 0))
                shots = int(get_stat('shots', 0))
                shots_on_target = int(get_stat('shots_on_target', 0))
                xg = get_stat('xg', 0.0)
                npxg = get_stat('npxg', 0.0)  # Non-penalty xG
                xg_per_shot = get_stat('xg_per_shot', 0.0)

                profile = TeamXGProfile(
                    team_name=team_name,
                    league=league,
                    season="2025-2026",
                    total_xg_for=xg,
                    total_goals_for=goals,
                    avg_xg_for=xg / max(matches_played, 1),
                    goals_minus_xg=goals - xg,
                    matches_played=matches_played,
                    data_available=True,
                    last_updated=datetime.now().isoformat()
                )

                profiles[team_name] = profile

            except Exception as e:
                continue

        print(f"      ✅ {len(profiles)} équipes chargées depuis FBref ({league})")
        return profiles

    # ─── xG D'UN MATCH SPÉCIFIQUE ───────────────────

    def get_match_xg(self, match_url: str) -> Optional[MatchXG]:
        """
        Récupère les xG détaillés d'un match spécifique via son URL FBref.
        """

        soup = self._fetch_page(match_url)
        if not soup:
            return None

        try:
            # Extraire les équipes
            scorebox = soup.find('div', class_='scorebox')
            if not scorebox:
                return None

            teams = scorebox.find_all('strong')
            team_links = scorebox.find_all('a', {'itemprop': 'name'})

            if len(team_links) >= 2:
                home_team = team_links[0].get_text().strip()
                away_team = team_links[1].get_text().strip()
            else:
                return None

            # Score
            scores = scorebox.find_all('div', class_='score')
            home_goals = int(scores[0].get_text()) if len(scores) >= 1 else 0
            away_goals = int(scores[1].get_text()) if len(scores) >= 2 else 0

            # xG (dans les divs sous le score)
            xg_divs = scorebox.find_all('div', class_='score_xg')
            home_xg = 0.0
            away_xg = 0.0

            if len(xg_divs) >= 2:
                try:
                    home_xg = float(xg_divs[0].get_text().strip())
                    away_xg = float(xg_divs[1].get_text().strip())
                except (ValueError, AttributeError):
                    pass

            match = MatchXG(
                date=datetime.now().strftime("%Y-%m-%d"),
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_xg=home_xg,
                away_xg=away_xg,
                source="fbref"
            )

            return match

        except Exception as e:
            print(f"      ❌ Erreur parsing match: {e}")
            return None

    # ─── MATCHS RÉCENTS D'UNE ÉQUIPE ────────────────

    def get_team_recent_xg(self, team_url: str,
                            last_n: int = 10) -> List[MatchXG]:
        """
        Récupère les xG des N derniers matchs d'une équipe.
        """

        # Construire l'URL des matchs
        if '/squads/' in team_url:
            schedule_url = team_url.replace('/squads/', '/squads/')
        else:
            schedule_url = team_url

        soup = self._fetch_page(schedule_url)
        if not soup:
            return []

        matches = []

        # Trouver le tableau des scores & fixtures
        table = soup.find('table', {'id': 'matchlogs_for'})
        if not table:
            tables = soup.find_all('table')
            for t in tables:
                caption = t.find('caption')
                if caption and 'Scores' in caption.get_text():
                    table = t
                    break

        if not table:
            return []

        tbody = table.find('tbody')
        if not tbody:
            return []

        rows = tbody.find_all('tr')

        for row in rows[-last_n:]:
            if row.get('class') and 'thead' in ' '.join(row.get('class', [])):
                continue

            try:
                def get_cell(stat_name):
                    cell = row.find('td', {'data-stat': stat_name})
                    return cell.get_text().strip() if cell else ""

                date = get_cell('date')
                venue = get_cell('venue')
                opponent = get_cell('opponent')
                goals_for = get_cell('goals_for')
                goals_against = get_cell('goals_against')
                xg_for = get_cell('xg')
                xg_against = get_cell('xg_against')

                if not date or not opponent:
                    continue

                match = MatchXG(
                    date=date,
                    home_team="" if venue == "Away" else opponent,
                    away_team=opponent if venue == "Home" else "",
                    home_goals=int(goals_for) if goals_for else 0,
                    away_goals=int(goals_against) if goals_against else 0,
                    home_xg=float(xg_for) if xg_for else 0.0,
                    away_xg=float(xg_against) if xg_against else 0.0,
                    source="fbref"
                )

                matches.append(match)

            except (ValueError, AttributeError):
                continue

        return matches


# ══════════════════════════════════════════════════════
#  SCRAPER UNDERSTAT (xG détaillés)
# ══════════════════════════════════════════════════════

class UnderstatScraper:
    """
    Scrape les données xG depuis Understat.com.

    Understat offre des xG par tir, par joueur et par match
    pour les 6 ligues majeures :
    EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL
    """

    BASE_URL = "https://understat.com"

    LEAGUE_NAMES = {
        "premier_league": "EPL",
        "la_liga": "La_liga",
        "serie_a": "Serie_A",
        "bundesliga": "Bundesliga",
        "ligue1_fr": "Ligue_1",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
            )
        })
        self._request_count = 0

    def _fetch_page(self, url: str) -> Optional[str]:
        """Récupère le HTML brut d'une page Understat."""

        self._request_count += 1
        if self._request_count > 1:
            time.sleep(3)

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                print(f"      ❌ HTTP {response.status_code} pour {url}")
                return None
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
            return None

    def _extract_json_var(self, html: str, var_name: str) -> Optional[list]:
        """
        Understat stocke les données dans des variables JavaScript.
        Cette méthode extrait le JSON depuis le code JS.

        Exemple : var teamsData = JSON.parse('...');
        """

        pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)

        if match:
            json_str = match.group(1)
            # Décoder les caractères échappés
            json_str = json_str.encode().decode('unicode_escape')
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None

        return None

    # ─── xG PAR ÉQUIPE SUR LA SAISON ────────────────

    def get_league_teams_xg(self, league: str,
                             season: int = 2025) -> Dict[str, TeamXGProfile]:
        """
        Récupère les xG de toutes les équipes d'une ligue sur la saison.
        """

        league_name = self.LEAGUE_NAMES.get(league)
        if not league_name:
            print(f"      ⚠️ Ligue '{league}' non supportée par Understat")
            return {}

        url = f"{self.BASE_URL}/league/{league_name}/{season}"
        html = self._fetch_page(url)

        if not html:
            return {}

        # Extraire teamsData du JavaScript
        teams_data = self._extract_json_var(html, 'teamsData')

        if not teams_data:
            print(f"      ⚠️ Données d'équipe non trouvées sur Understat")
            return {}

        profiles = {}

        for team_id, team_info in teams_data.items():
            team_name = team_info.get('title', 'Unknown')

            # Données historiques par match
            history = team_info.get('history', [])

            if not history:
                continue

            total_xg_for = 0.0
            total_xg_against = 0.0
            total_goals_for = 0
            total_goals_against = 0

            home_xg_for = []
            home_xg_against = []
            away_xg_for = []
            away_xg_against = []

            recent_xg_for = []
            recent_xg_against = []

            for match in history:
                xg_f = float(match.get('xG', 0))
                xg_a = float(match.get('xGA', 0))
                goals_f = int(match.get('scored', 0))
                goals_a = int(match.get('missed', 0))  # 'missed' = buts encaissés
                is_home = match.get('h_a', '') == 'h'

                total_xg_for += xg_f
                total_xg_against += xg_a
                total_goals_for += goals_f
                total_goals_against += goals_a

                if is_home:
                    home_xg_for.append(xg_f)
                    home_xg_against.append(xg_a)
                else:
                    away_xg_for.append(xg_f)
                    away_xg_against.append(xg_a)

            # 5 derniers matchs
            for match in history[-5:]:
                recent_xg_for.append(float(match.get('xG', 0)))
                recent_xg_against.append(float(match.get('xGA', 0)))

            mp = len(history)

            profile = TeamXGProfile(
                team_name=team_name,
                league=league,
                season=f"{season}-{season+1}",

                total_xg_for=round(total_xg_for, 2),
                total_xg_against=round(total_xg_against, 2),
                total_goals_for=total_goals_for,
                total_goals_against=total_goals_against,

                avg_xg_for=round(total_xg_for / max(mp, 1), 3),
                avg_xg_against=round(total_xg_against / max(mp, 1), 3),

                xg_difference=round(total_xg_for - total_xg_against, 2),
                goals_minus_xg=round(total_goals_for - total_xg_for, 2),
                goals_against_minus_xga=round(
                    total_goals_against - total_xg_against, 2
                ),

                recent_xg_for=recent_xg_for,
                recent_xg_against=recent_xg_against,
                recent_avg_xg_for=round(
                    sum(recent_xg_for) / max(len(recent_xg_for), 1), 3
                ),
                recent_avg_xg_against=round(
                    sum(recent_xg_against) / max(len(recent_xg_against), 1), 3
                ),

                home_avg_xg_for=round(
                    sum(home_xg_for) / max(len(home_xg_for), 1), 3
                ),
                home_avg_xg_against=round(
                    sum(home_xg_against) / max(len(home_xg_against), 1), 3
                ),
                away_avg_xg_for=round(
                    sum(away_xg_for) / max(len(away_xg_for), 1), 3
                ),
                away_avg_xg_against=round(
                    sum(away_xg_against) / max(len(away_xg_against), 1), 3
                ),

                matches_played=mp,
                data_available=True,
                last_updated=datetime.now().isoformat()
            )

            profiles[team_name] = profile

        print(f"      ✅ {len(profiles)} équipes chargées depuis Understat ({league})")
        return profiles

    # ─── xG D'UN MATCH SPÉCIFIQUE ───────────────────

    def get_match_detail_xg(self, match_id: str) -> Optional[Dict]:
        """
        Récupère les xG détaillés tir par tir d'un match.

        Args:
            match_id: ID du match sur Understat (ex: "12345")
        """

        url = f"{self.BASE_URL}/match/{match_id}"
        html = self._fetch_page(url)

        if not html:
            return None

        # Extraire shotsData
        shots_data = self._extract_json_var(html, 'shotsData')
        match_info = self._extract_json_var(html, 'match_info')

        if not shots_data:
            return None

        result = {
            "match_id": match_id,
            "home_shots": [],
            "away_shots": [],
            "home_total_xg": 0.0,
            "away_total_xg": 0.0,
        }

        # Parser les tirs
        for side in ['h', 'a']:
            side_shots = shots_data.get(side, [])
            for shot in side_shots:
                shot_info = {
                    "player": shot.get('player', ''),
                    "minute": int(shot.get('minute', 0)),
                    "xg": float(shot.get('xG', 0)),
                    "result": shot.get('result', ''),
                    "situation": shot.get('situation', ''),
                    "shot_type": shot.get('shotType', ''),
                    "x": float(shot.get('X', 0)),
                    "y": float(shot.get('Y', 0)),
                }

                if side == 'h':
                    result["home_shots"].append(shot_info)
                    result["home_total_xg"] += shot_info["xg"]
                else:
                    result["away_shots"].append(shot_info)
                    result["away_total_xg"] += shot_info["xg"]

        return result


# ══════════════════════════════════════════════════════
#  ORCHESTRATEUR xG
# ══════════════════════════════════════════════════════

class XGCollector:
    """
    Orchestrateur qui combine FBref et Understat pour obtenir
    les données xG les plus complètes possibles.

    Priorité :
    1. Understat (données JSON structurées, plus facile à parser)
    2. FBref (fallback, données StatsBomb très fiables)
    """

    def __init__(self):
        self.understat = UnderstatScraper()
        self.fbref = FBrefScraper()
        self._cache = {}

    def get_team_xg_profile(self, team_name: str,
                             league: str) -> Optional[TeamXGProfile]:
        """
        Récupère le profil xG complet d'une équipe.

        Essaie d'abord Understat, puis FBref en fallback.
        """

        cache_key = f"{league}_{team_name}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Essayer de charger la ligue entière (plus efficace)
        league_cache_key = f"league_{league}"

        if league_cache_key not in self._cache:
            print(f"    📊 Chargement des xG pour {league}...")

            # Essayer Understat d'abord
            profiles = self.understat.get_league_teams_xg(league)

            if not profiles:
                # Fallback FBref
                profiles = self.fbref.get_league_xg_table(league)

            if profiles:
                self._cache[league_cache_key] = profiles
                for name, profile in profiles.items():
                    self._cache[f"{league}_{name}"] = profile
            else:
                self._cache[league_cache_key] = {}

        # Chercher l'équipe dans le cache
        result = self._cache.get(cache_key)

        if not result:
            # Fuzzy search dans les profils de la ligue
            league_profiles = self._cache.get(league_cache_key, {})
            from rapidfuzz import fuzz, process

            if league_profiles:
                match = process.extractOne(
                    team_name,
                    list(league_profiles.keys()),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=70
                )

                if match:
                    matched_name = match[0]
                    result = league_profiles[matched_name]
                    self._cache[cache_key] = result

        return result

    def enrich_match_context(self, context, home_team: str,
                              away_team: str, league: str):
        """
        Enrichit un MatchContext avec les données xG.

        Met à jour les TeamStats avec les xG si disponibles.
        """

        home_xg = self.get_team_xg_profile(home_team, league)
        away_xg = self.get_team_xg_profile(away_team, league)

        if home_xg and context.home_stats:
            context.home_stats.xg_scored = home_xg.avg_xg_for
            context.home_stats.xg_conceded = home_xg.avg_xg_against
            context.home_stats.xg_available = True
            print(f"      📈 xG {home_team}: {home_xg.avg_xg_for:.2f} pour "
                  f"/ {home_xg.avg_xg_against:.2f} contre")

        if away_xg and context.away_stats:
            context.away_stats.xg_scored = away_xg.avg_xg_for
            context.away_stats.xg_conceded = away_xg.avg_xg_against
            context.away_stats.xg_available = True
            print(f"      📈 xG {away_team}: {away_xg.avg_xg_for:.2f} pour "
                  f"/ {away_xg.avg_xg_against:.2f} contre")

        if home_xg and away_xg:
            context.data_completeness = min(context.data_completeness + 15, 100)

        return context, home_xg, away_xg
