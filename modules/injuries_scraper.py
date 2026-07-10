"""
═══════════════════════════════════════════════════════
 MODULE INJURIES — Scraping des blessures et
 suspensions depuis Transfermarkt
═══════════════════════════════════════════════════════
"""

import requests
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from bs4 import BeautifulSoup


@dataclass
class PlayerAbsence:
    """Un joueur absent (blessé ou suspendu)."""

    name: str
    team: str
    reason: str          # "Blessure au genou", "Carton rouge", etc.
    absence_type: str    # "injury", "suspension", "international"
    expected_return: str  # Date de retour estimée
    market_value: float  # Valeur marchande (millions €)
    position: str        # "Attaquant", "Milieu", "Défenseur", "Gardien"
    is_key_player: bool  # Joueur titulaire régulier

    # Impact estimé
    impact_score: float = 0.0  # 0 (aucun) à 1.0 (critique)


@dataclass
class TeamAbsences:
    """Toutes les absences d'une équipe."""

    team_name: str
    total_absent: int = 0
    injuries: List[PlayerAbsence] = field(default_factory=list)
    suspensions: List[PlayerAbsence] = field(default_factory=list)

    # Impact global
    total_impact: float = 0.0
    key_players_absent: int = 0

    # Détail par ligne
    attackers_absent: int = 0
    midfielders_absent: int = 0
    defenders_absent: int = 0
    goalkeepers_absent: int = 0


class InjuriesScraper:
    """
    Scrape les informations de blessures et suspensions
    depuis Transfermarkt et d'autres sources.

    Impact typique des absences :
    - Attaquant star : -8 à -12% sur prob de victoire
    - Gardien titulaire : -5 à -8%
    - Défenseur central : -3 à -6%
    - Milieu clé : -4 à -7%
    """

    TRANSFERMARKT_BASE = "https://www.transfermarkt.com"

    # Impact par position et importance
    IMPACT_WEIGHTS = {
        "Attaquant": {"key": 0.12, "regular": 0.06, "rotation": 0.02},
        "Milieu": {"key": 0.08, "regular": 0.04, "rotation": 0.01},
        "Défenseur": {"key": 0.07, "regular": 0.03, "rotation": 0.01},
        "Gardien": {"key": 0.10, "regular": 0.05, "rotation": 0.01},
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'fr-FR,fr;q=0.9'
        })

    def get_team_absences(self, team_name: str,
                          team_url: str = None) -> Optional[TeamAbsences]:
        """
        Récupère les absences d'une équipe.

        Si team_url n'est pas fourni, cherche l'équipe sur Transfermarkt.
        """

        absences = TeamAbsences(team_name=team_name)

        if not team_url:
            team_url = self._search_team(team_name)
            if not team_url:
                return absences

        try:
            # Construire l'URL des blessures
            injury_url = team_url.replace("/startseite/", "/sperrenundverletzungen/")

            time.sleep(3)  # Rate limit

            response = self.session.get(
                f"{self.TRANSFERMARKT_BASE}{injury_url}",
                timeout=20
            )

            if response.status_code != 200:
                return absences

            soup = BeautifulSoup(response.text, 'html.parser')

            # Parser le tableau des blessures
            injury_table = soup.find('table', class_='items')

            if not injury_table:
                return absences

            rows = injury_table.find_all('tr', class_=['odd', 'even'])

            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 4:
                    continue

                try:
                    # Nom du joueur
                    name_cell = row.find('td', class_='hauptlink')
                    name = name_cell.get_text().strip() if name_cell else "Inconnu"

                    # Position
                    pos_cell = cells[1] if len(cells) > 1 else None
                    position_raw = pos_cell.get_text().strip() if pos_cell else ""
                    position = self._normalize_position(position_raw)

                    # Raison
                    reason_cell = cells[-2] if len(cells) > 2 else None
                    reason = reason_cell.get_text().strip() if reason_cell else "Inconnu"

                    # Type
                    absence_type = "suspension" if "carton" in reason.lower() or "susp" in reason.lower() else "injury"

                    # Valeur marchande
                    value = 0.0
                    value_cell = row.find('td', class_='rechts')
                    if value_cell:
                        value_text = value_cell.get_text().strip()
                        value = self._parse_market_value(value_text)

                    player = PlayerAbsence(
                        name=name,
                        team=team_name,
                        reason=reason,
                        absence_type=absence_type,
                        expected_return="",
                        market_value=value,
                        position=position,
                        is_key_player=value > 5.0  # > 5M€ = joueur clé
                    )

                    # Calculer l'impact
                    importance = "key" if player.is_key_player else "regular"
                    weights = self.IMPACT_WEIGHTS.get(position, {})
                    player.impact_score = weights.get(importance, 0.03)

                    if absence_type == "injury":
                        absences.injuries.append(player)
                    else:
                        absences.suspensions.append(player)

                    # Compter par ligne
                    if position == "Attaquant":
                        absences.attackers_absent += 1
                    elif position == "Milieu":
                        absences.midfielders_absent += 1
                    elif position == "Défenseur":
                        absences.defenders_absent += 1
                    elif position == "Gardien":
                        absences.goalkeepers_absent += 1

                    if player.is_key_player:
                        absences.key_players_absent += 1

                except Exception:
                    continue

            absences.total_absent = len(absences.injuries) + len(absences.suspensions)
            absences.total_impact = sum(
                p.impact_score for p in absences.injuries + absences.suspensions
            )

            return absences

        except Exception as e:
            print(f"      ⚠️ Erreur scraping blessures: {e}")
            return absences

    def _search_team(self, team_name: str) -> Optional[str]:
        """Recherche une équipe sur Transfermarkt."""

        try:
            response = self.session.get(
                f"{self.TRANSFERMARKT_BASE}/schnellsuche/ergebnis/schnellsuche",
                params={"query": team_name},
                timeout=15
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                result = soup.find('a', class_='vereinprofil_tooltip')
                if result:
                    return result.get('href')

            return None

        except Exception:
            return None

    def _normalize_position(self, pos: str) -> str:
        """Normalise le nom de position."""

        pos_lower = pos.lower()

        if any(w in pos_lower for w in ['attaq', 'avant', 'ailier', 'buteur', 'forward', 'striker', 'winger']):
            return "Attaquant"
        elif any(w in pos_lower for w in ['milieu', 'midfield', 'meneur', 'relayeur']):
            return "Milieu"
        elif any(w in pos_lower for w in ['défen', 'arrière', 'latéral', 'defender', 'back']):
            return "Défenseur"
        elif any(w in pos_lower for w in ['gardien', 'keeper', 'goal']):
            return "Gardien"

        return "Milieu"  # Default

    def _parse_market_value(self, value_str: str) -> float:
        """Parse une valeur marchande Transfermarkt en millions €."""

        try:
            value_str = value_str.strip().lower()

            if 'mio' in value_str or 'mill' in value_str or 'm' in value_str:
                number = ''.join(c for c in value_str if c.isdigit() or c in '.,')
                number = number.replace(',', '.')
                return float(number)
            elif 'tsd' in value_str or 'k' in value_str:
                number = ''.join(c for c in value_str if c.isdigit() or c in '.,')
                number = number.replace(',', '.')
                return float(number) / 1000

            return 0.0
        except (ValueError, AttributeError):
            return 0.0

    def calculate_absence_impact(self, absences: TeamAbsences) -> Dict:
        """
        Calcule l'impact global des absences sur les probabilités.

        Returns:
            Dict avec les ajustements de probabilité
        """

        total_impact = absences.total_impact

        return {
            "win_prob_adjustment": -total_impact,  # Réduction de la prob de victoire
            "goals_adjustment": -total_impact * 0.3,  # Réduction des buts attendus
            "description": self._describe_impact(absences),
            "severity": (
                "🔴 Critique" if total_impact > 0.15 else
                "🟡 Modéré" if total_impact > 0.05 else
                "🟢 Faible"
            )
        }

    def _describe_impact(self, absences: TeamAbsences) -> str:
        """Génère une description textuelle de l'impact."""

        parts = []

        if absences.total_absent == 0:
            return "Effectif au complet"

        parts.append(f"{absences.total_absent} absent(s)")

        if absences.key_players_absent > 0:
            key_names = [
                p.name for p in absences.injuries + absences.suspensions
                if p.is_key_player
            ][:3]
            parts.append(f"Joueurs clés absents: {', '.join(key_names)}")

        if absences.attackers_absent > 0:
            parts.append(f"{absences.attackers_absent} attaquant(s) absent(s)")

        return " | ".join(parts)
