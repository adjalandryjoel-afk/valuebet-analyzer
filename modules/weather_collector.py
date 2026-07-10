"""
═══════════════════════════════════════════════════════
 MODULE WEATHER — Collecte des conditions météo
 pour le jour et l'heure du match
═══════════════════════════════════════════════════════
"""

import requests
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

from config import APIKeys


@dataclass
class MatchWeather:
    """Conditions météo pour un match."""

    city: str
    temperature: float = 0.0       # °C
    feels_like: float = 0.0
    humidity: int = 0              # %
    wind_speed: float = 0.0        # km/h
    wind_direction: str = ""
    description: str = ""
    rain_probability: float = 0.0  # %
    rain_mm: float = 0.0           # mm de pluie prévue

    # Impact estimé sur le jeu
    impact_score: float = 0.0      # -1 (défavorable) à +1 (favorable)
    impact_description: str = ""

    # Recommandations
    favors_under: bool = False     # Conditions qui favorisent Under
    favors_over: bool = False


class WeatherCollector:
    """
    Collecte les conditions météo via OpenWeatherMap API.

    Impact de la météo sur le football :
    - Pluie forte : réduit la qualité du jeu, favorise Under
    - Vent fort (>30 km/h) : pertube les centres et tirs, favorise Under
    - Chaleur extrême (>35°C) : fatigue, rythme plus lent
    - Froid extrême (<5°C) : terrain dur, blessures
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    # Coordonnées des villes de stades courants
    CITY_COORDS = {
        "abidjan": (5.3364, -4.0267),
        "paris": (48.8566, 2.3522),
        "marseille": (43.2965, 5.3698),
        "lyon": (45.7640, 4.8357),
        "london": (51.5074, -0.1278),
        "manchester": (53.4808, -2.2426),
        "liverpool": (53.4084, -2.9916),
        "madrid": (40.4168, -3.7038),
        "barcelona": (41.3874, 2.1686),
        "munich": (48.1351, 11.5820),
        "milan": (45.4642, 9.1900),
        "rome": (41.9028, 12.4964),
        "san-pédro": (4.7485, -6.6363),
        "bouaké": (7.6939, -5.0308),
        "gagnoa": (6.1319, -5.9506),
    }

    def __init__(self):
        self.api_key = APIKeys.OPENAI_API_KEY  # Utiliser une clé dédiée en prod

    def get_weather(self, city: str,
                    match_date: str = None) -> Optional[MatchWeather]:
        """
        Récupère la météo pour une ville donnée.

        Args:
            city: Nom de la ville
            match_date: Date du match (YYYY-MM-DD), utilise aujourd'hui si None
        """

        city_lower = city.lower().strip()
        coords = self.CITY_COORDS.get(city_lower)

        try:
            if coords:
                lat, lon = coords
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "metric",
                    "lang": "fr"
                }
            else:
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": "metric",
                    "lang": "fr"
                }

            response = requests.get(
                f"{self.BASE_URL}/weather",
                params=params,
                timeout=10
            )

            if response.status_code != 200:
                return None

            data = response.json()

            weather = MatchWeather(
                city=city,
                temperature=data["main"]["temp"],
                feels_like=data["main"]["feels_like"],
                humidity=data["main"]["humidity"],
                wind_speed=data["wind"]["speed"] * 3.6,  # m/s → km/h
                description=data["weather"][0]["description"],
                rain_mm=data.get("rain", {}).get("1h", 0.0)
            )

            # Calculer l'impact
            self._calculate_impact(weather)

            return weather

        except Exception as e:
            print(f"      ⚠️ Météo indisponible pour {city}: {e}")
            return None

    def _calculate_impact(self, weather: MatchWeather):
        """Calcule l'impact de la météo sur le match."""

        impact = 0.0
        descriptions = []

        # Pluie
        if weather.rain_mm > 5:
            impact -= 0.4
            weather.favors_under = True
            descriptions.append("Forte pluie → Under favorisé")
        elif weather.rain_mm > 1:
            impact -= 0.2
            descriptions.append("Pluie modérée")

        # Vent
        if weather.wind_speed > 40:
            impact -= 0.5
            weather.favors_under = True
            descriptions.append("Vent très fort → Under favorisé")
        elif weather.wind_speed > 25:
            impact -= 0.2
            descriptions.append("Vent notable")

        # Température
        if weather.temperature > 35:
            impact -= 0.3
            descriptions.append("Chaleur extrême → rythme réduit")
        elif weather.temperature < 3:
            impact -= 0.1
            descriptions.append("Froid → terrain dur")
        elif 15 <= weather.temperature <= 25:
            impact += 0.1
            descriptions.append("Conditions idéales")

        # Humidité
        if weather.humidity > 85:
            impact -= 0.1
            descriptions.append("Humidité élevée")

        weather.impact_score = max(-1, min(1, impact))
        weather.impact_description = " | ".join(descriptions) if descriptions else "Conditions normales"

        if not weather.favors_under and impact > -0.1:
            weather.favors_over = True
