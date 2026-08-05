"""
═══════════════════════════════════════════════════════
 MODULE MATCH BRIEF — la fiche factuelle d'un match
═══════════════════════════════════════════════════════

Rassemble automatiquement ce que l'utilisateur cherchait à la main
avant chaque pari : qui manque, qui est suspendu, qui sort d'un match
il y a trois jours, ce que chaque équipe joue au classement, et le
temps qu'il fera.

┌─ CE MODULE N'AJUSTE AUCUNE PROBABILITÉ ─────────────────────────┐
│                                                                  │
│ Il AFFICHE des faits. Il ne les envoie pas à l'agent contexte    │
│ et ne touche pas aux multiplicateurs de λ.                       │
│                                                                  │
│ Raison : l'agent contexte peut déplacer λ de ±12 %, il est de    │
│ loin le plus puissant de l'app, et ce ±12 % n'est calibré par    │
│ RIEN. Aujourd'hui il ne tourne que si l'utilisateur écrit        │
│ quelque chose (match_intel.py, court-circuit « notes vides »).   │
│ Le brancher automatiquement l'allumerait sur tous les matchs,    │
│ en permanence — et invaliderait le seul chiffre mesuré du        │
│ projet, le CLV de −5 %, obtenu précisément avec cet agent au     │
│ repos.                                                           │
│                                                                  │
│ L'utilisateur lit la fiche et décide lui-même de ce qu'il écrit  │
│ dans ses notes. C'est le travail de RECHERCHE qui est            │
│ automatisé, pas le jugement.                                     │
└──────────────────────────────────────────────────────────────────┘

RÈGLE ABSOLUE DE CE MODULE : ne jamais transformer une absence de
donnée en affirmation. « Aucun absent listé » n'est PAS « effectif au
complet » — la sous-déclaration de la source est prouvée (un joueur
absent de cinq feuilles de match consécutives n'avait aucune ligne).
Chaque bloc porte donc un drapeau `dispo` distinct de son contenu.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from modules.api_football import get_api_collector

# ── Ligue interne → identifiant API-Football ────────────
# Vérifiée compétition par compétition : chaque identifiant a bien
# renvoyé les matchs du bon championnat.
LEAGUE_IDS = {
    "premier_league": 39, "la_liga": 140, "serie_a": 135,
    "bundesliga": 78, "ligue1_fr": 61, "eredivisie": 88,
    "primeira_liga": 94, "belgium_pro": 144, "super_lig": 203,
    "greece_super": 197, "scotland_prem": 179,
    "swe_allsvenskan": 113, "nor_eliteserien": 103,
    "fin_veikkaus": 244, "den_superliga": 119,
    "aut_bundesliga": 218, "pol_ekstraklasa": 106,
    "rus_premier": 235, "championship": 40, "bundesliga2": 79,
    "champions_league": 2, "europa_league": 3,
}

# Compétitions à EXCLURE du calcul de repos et de charge.
# 667 = « Friendlies Clubs ». Sans ce filtre, le dernier match du PSG
# est un amical de préparation et l'app annonce 66 jours de repos.
LIGUES_AMICALES = {667, 10, 666}

# Statuts d'un match réellement joué
STATUTS_JOUES = ("FT", "AET", "PEN")

# En dessous de ce nombre de journées, un classement ne dit rien :
# à zéro journée, l'API renvoie les équipes par ORDRE ALPHABÉTIQUE
# avec le champ « description » déjà rempli — de quoi annoncer
# sérieusement « Auxerre 1er, place qualificative Ligue des Champions ».
MIN_JOURNEES_CLASSEMENT = 5

# Traduction des motifs d'absence (l'API répond en anglais)
MOTIFS_FR = {
    "injury": "blessure", "knee injury": "genou",
    "thigh injury": "cuisse", "muscle injury": "muscle",
    "ankle injury": "cheville", "back injury": "dos",
    "groin injury": "aine", "calf injury": "mollet",
    "hamstring injury": "ischio-jambiers",
    "shoulder injury": "épaule", "foot injury": "pied",
    "broken leg": "jambe cassée", "achilles tendon injury": "tendon d'Achille",
    "illness": "maladie", "suspended": "suspendu",
    "red card": "carton rouge", "yellow cards": "cumul de cartons jaunes",
    "international duty": "sélection nationale",
    "coach decision": "choix de l'entraîneur", "inactive": "inactif",
    "rest": "mise au repos", "personal reasons": "raisons personnelles",
    "knock": "coup reçu", "hip injury": "hanche",
    "toe injury": "orteil", "head injury": "tête",
    "abdominal injury": "abdominaux", "elbow injury": "coude",
    "wrist injury": "poignet", "fitness": "condition physique",
    "unknown": "motif inconnu", "national selection": "sélection nationale",
    "cruciate ligament injury": "ligaments croisés",
    "ankle/foot injury": "cheville/pied", "knee/ankle injury": "genou/cheville",
    "muscular problems": "problèmes musculaires",
    "shin injury": "tibia", "face injury": "visage",
    "adductor problems": "adducteurs", "fatigue": "fatigue",
}

# Au-delà de ce nombre de jours sans match, ce n'est plus du « repos »
# mais une REPRISE (intersaison, blessure longue, promotion). Afficher
# « 105 jours de repos » comme un avantage de fraîcheur serait un
# contresens : une équipe qui n'a pas joué depuis trois mois n'est pas
# reposée, elle est sans rythme.
SEUIL_REPRISE_JOURS = 30


def _fr_motif(motif: Optional[str]) -> str:
    if not motif:
        return "motif non précisé"
    return MOTIFS_FR.get(str(motif).strip().lower(), str(motif))


# ══════════════════════════════════════════════════════
#  STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class Absent:
    nom: str
    motif: str
    certain: bool   # False = « incertain » (Questionable) côté API


@dataclass
class FicheEquipe:
    """Les faits connus d'une équipe pour ce match."""

    nom: str

    # ─ Calendrier ─
    repos_jours: Optional[float] = None
    reprise: bool = False                  # long arrêt, pas du repos
    dernier_match: Optional[Dict] = None   # date, adversaire, compet, score
    prolongations: bool = False
    matchs_14j: Optional[int] = None
    match_suivant: Optional[Dict] = None   # pour repérer un « sandwich »

    # ─ Absents ─ (dispo distinct du contenu : voir en-tête du module)
    absents: List[Absent] = field(default_factory=list)
    absents_dispo: bool = False

    # ─ Suspensions probables ─
    rouges: List[str] = field(default_factory=list)
    rouges_dispo: bool = False

    # ─ Classement ─
    rang: Optional[int] = None
    points: Optional[int] = None
    journees: Optional[int] = None
    enjeu: Optional[str] = None            # « Relegation - … », traduit
    classement_fiable: bool = False


@dataclass
class MatchBrief:
    """Fiche complète d'un match."""

    domicile: FicheEquipe
    exterieur: FicheEquipe
    kickoff: Optional[datetime] = None
    meteo: Optional[Dict] = None
    stade: Optional[str] = None
    avertissements: List[str] = field(default_factory=list)
    requetes: int = 0

    def est_vide(self) -> bool:
        """True si rien d'exploitable n'a pu être réuni."""
        for e in (self.domicile, self.exterieur):
            if (e.repos_jours is not None or e.absents
                    or e.rouges or e.rang is not None):
                return False
        return self.meteo is None

    def resume(self) -> str:
        """
        Résumé en français, prêt à être lu — ou recopié dans les notes
        si l'utilisateur juge qu'un point mérite d'entrer dans
        l'analyse. C'est LUI qui décide : ce texte n'est jamais envoyé
        au modèle automatiquement (voir l'en-tête du module).
        """

        lignes = []
        for role, eq in (("Domicile", self.domicile),
                         ("Extérieur", self.exterieur)):
            bouts = []

            if eq.reprise and eq.repos_jours is not None:
                bouts.append(f"reprise après {eq.repos_jours:.0f} jours "
                             f"sans match")
            elif eq.repos_jours is not None:
                bouts.append(f"{eq.repos_jours:.0f} jours de repos")
            if eq.prolongations:
                bouts.append("a joué les prolongations")
            if eq.matchs_14j and eq.matchs_14j >= 3:
                bouts.append(f"{eq.matchs_14j} matchs en 14 jours")

            certains = [a for a in eq.absents if a.certain]
            doutes = [a for a in eq.absents if not a.certain]
            if certains:
                bouts.append("absents : " + ", ".join(
                    f"{a.nom} ({a.motif})" for a in certains))
            if doutes:
                bouts.append("incertains : " + ", ".join(
                    f"{a.nom} ({a.motif})" for a in doutes))
            if eq.absents_dispo and not eq.absents:
                bouts.append("aucun absent listé (la source couvre ce "
                             "match, mais elle sous-déclare)")
            if not eq.absents_dispo:
                bouts.append("absents inconnus (aucune donnée)")

            if eq.rouges:
                bouts.append("rouge au dernier match, suspension "
                             "probable : " + ", ".join(eq.rouges))

            if eq.classement_fiable and eq.rang:
                bout = f"{eq.rang}e avec {eq.points} points"
                if eq.enjeu:
                    bout += f" ({eq.enjeu})"
                bouts.append(bout)

            if eq.match_suivant and (eq.match_suivant.get("dans_jours")
                                     or 99) <= 4:
                bouts.append(
                    f"rejoue {eq.match_suivant['competition']} "
                    f"{eq.match_suivant['dans_jours']} jours après")

            if bouts:
                lignes.append(f"{role} ({eq.nom}) — " + " ; ".join(bouts))

        if self.meteo:
            m = self.meteo
            lignes.append(
                f"Météo à {m['ville']} : {m['temperature']} °C, "
                f"{m['pluie_mm']} mm de pluie, vent {m['vent_kmh']} km/h")

        return "\n".join(lignes)


# ══════════════════════════════════════════════════════
#  CONSTRUCTION
# ══════════════════════════════════════════════════════

class MatchBriefBuilder:
    """
    Construit la fiche d'un match. Chaque bloc est indépendant et
    défensif : s'il échoue, il laisse son drapeau `dispo` à False et
    les autres continuent. Aucune exception ne remonte.
    """

    # Durée de vie d'une fiche en mémoire. Courte : une liste
    # d'absents peut changer dans la journée. Sert surtout à éviter
    # de re-dépenser ~8 requêtes à CHAQUE réexécution du script
    # Streamlit (un simple clic en déclenche une).
    CACHE_TTL_S = 30 * 60

    def __init__(self):
        self.api = get_api_collector()
        self._standings_cache: Dict[Tuple[int, int], Dict] = {}
        self._briefs: Dict[Tuple, Tuple[float, MatchBrief]] = {}

    # ─── POINT D'ENTRÉE ─────────────────────────────

    def build(self, home: str, away: str,
              league: str = "", kickoff: Optional[datetime] = None
              ) -> MatchBrief:
        """
        Fiche du match `home` vs `away`.

        league : clé interne (premier_league…) — sert au classement.
        kickoff : coup d'envoi connu (sinon déduit du calendrier API).
        """

        cle = (home.lower().strip(), away.lower().strip(), league)
        garde = self._briefs.get(cle)
        if garde and (time.time() - garde[0]) < self.CACHE_TTL_S:
            return garde[1]

        fiche = MatchBrief(domicile=FicheEquipe(nom=home),
                           exterieur=FicheEquipe(nom=away),
                           kickoff=kickoff)

        n0 = getattr(self.api, "_request_count", 0)
        try:
            self._construire(fiche, home, away, league)
        except Exception as e:                     # pragma: no cover
            fiche.avertissements.append(
                f"Fiche partielle : {type(e).__name__}")
        fiche.requetes = getattr(self.api, "_request_count", 0) - n0

        self._briefs[cle] = (time.time(), fiche)
        return fiche

    def _construire(self, fiche: MatchBrief, home: str, away: str,
                    league: str):

        id_home = self.api._resolve_team_id(home)
        id_away = self.api._resolve_team_id(away)

        if not id_home or not id_away:
            manquantes = [n for n, i in ((home, id_home), (away, id_away))
                          if not i]
            fiche.avertissements.append(
                "Équipe(s) non trouvée(s) côté API : "
                + ", ".join(manquantes))

        # Calendrier de chaque équipe (matchs joués + programmés)
        cal_home = self._calendrier(id_home) if id_home else ([], [])
        cal_away = self._calendrier(id_away) if id_away else ([], [])

        # Le match lui-même : on le cherche dans les matchs programmés
        # de l'équipe à domicile en filtrant sur l'adversaire.
        fixture = self._trouver_affiche(cal_home[1], id_away)
        if fixture and not fiche.kickoff:
            fiche.kickoff = _parse_dt(
                (fixture.get("fixture") or {}).get("date"))
        if fixture:
            venue = (fixture.get("fixture") or {}).get("venue") or {}
            fiche.stade = venue.get("name")

        ko = fiche.kickoff

        for eq, ident, cal in ((fiche.domicile, id_home, cal_home),
                               (fiche.exterieur, id_away, cal_away)):
            if not ident:
                continue
            self._remplir_calendrier(eq, cal, ko)
            self._remplir_rouges(eq, cal[0])

        # Absents : UNE requête donne les deux équipes
        if fixture:
            self._remplir_absents(fiche, fixture, id_home, id_away)
        elif cal_home[0] or cal_away[0]:
            fiche.avertissements.append(
                "Match introuvable au calendrier API — absents non "
                "consultés (l'affiche n'est peut-être pas encore "
                "programmée, ou les noms diffèrent)")

        # Classement
        if league in LEAGUE_IDS:
            self._remplir_classement(fiche, league, id_home, id_away)

        # Météo (gratuite, hors quota API-Football)
        if fixture and ko:
            fiche.meteo = self._meteo(fixture, ko)

    # ─── CALENDRIER ─────────────────────────────────

    def _calendrier(self, team_id: int) -> Tuple[List[Dict], List[Dict]]:
        """(matchs joués récents, matchs programmés) — 2 requêtes."""

        # `next` généreux : une équipe peut avoir plusieurs amicaux
        # ou un match de coupe avant l'affiche recherchée. Avec next=4,
        # Arsenal-Coventry (première journée) restait introuvable.
        # Le coût est le même : une requête, quel qu'en soit le nombre.
        joues = self._fixtures(team_id, {"last": 8}) or []
        prevus = self._fixtures(team_id, {"next": 12}) or []
        return joues, prevus

    def _fixtures(self, team_id: int, extra: Dict) -> Optional[List[Dict]]:
        params = {"team": team_id}
        params.update(extra)
        data = self.api._request("/fixtures", params)
        if not data or data.get("errors"):
            return None
        return data.get("response") or []

    @staticmethod
    def _trouver_affiche(prevus: List[Dict],
                         id_adverse: Optional[int]) -> Optional[Dict]:
        """Le prochain match opposant les deux équipes."""

        if not id_adverse:
            return None
        for fx in prevus:
            eq = fx.get("teams") or {}
            ids = {(eq.get("home") or {}).get("id"),
                   (eq.get("away") or {}).get("id")}
            if id_adverse in ids:
                return fx
        return None

    def _remplir_calendrier(self, eq: FicheEquipe,
                            cal: Tuple[List[Dict], List[Dict]],
                            ko: Optional[datetime]):
        """
        Repos réel et charge de calendrier.

        DEUX PIÈGES, tous deux mesurés :

        1. `last=N` ne renvoie que les matchs DÉJÀ JOUÉS. Un match
           programmé entre aujourd'hui et le coup d'envoi n'y figure
           pas — l'app annonçait 9,7 jours de repos à une équipe qui
           n'en avait que 2,8. On prend donc le plus récent des deux
           calendriers, tant qu'il précède le coup d'envoi.

        2. `last=N` inclut les AMICAUX. Sans filtre, le dernier match
           d'Arsenal est un amical et celui du PSG remonte à 66 jours.
        """

        joues, prevus = cal

        competitifs = [fx for fx in joues if _est_competitif(fx)
                       and _est_joue(fx)]
        if not competitifs:
            return

        competitifs.sort(key=_date_de, reverse=True)

        # Matchs programmés AVANT le coup d'envoi de celui qu'on analyse
        avant_ko = []
        if ko:
            avant_ko = [fx for fx in prevus
                        if _est_competitif(fx)
                        and (_date_de(fx) or "") < ko.isoformat()]
            avant_ko.sort(key=_date_de, reverse=True)

        dernier = competitifs[0]
        if avant_ko and (_date_de(avant_ko[0]) or "") > (_date_de(dernier) or ""):
            dernier = avant_ko[0]

        d_dt = _parse_dt(_date_de(dernier))
        eq.dernier_match = {
            "date": (_date_de(dernier) or "")[:10],
            "competition": ((dernier.get("league") or {}).get("name") or ""),
            "adversaire": _adversaire(dernier, eq.nom),
            "score": _score(dernier),
            "joue": _est_joue(dernier),
        }
        if ko and d_dt:
            eq.repos_jours = round(
                (ko - d_dt).total_seconds() / 86400.0, 1)
            eq.reprise = eq.repos_jours > SEUIL_REPRISE_JOURS

        # Les prolongations ne fatiguent que si le match est récent :
        # signaler une finale jouée il y a deux mois n'informe personne.
        statut = ((dernier.get("fixture") or {}).get("status") or {})
        eq.prolongations = (
            statut.get("short") in ("AET", "PEN")
            and eq.repos_jours is not None
            and eq.repos_jours <= 14)

        # Charge : matchs compétitifs joués dans les 14 jours avant le KO
        if ko:
            seuil = (ko.timestamp() - 14 * 86400)
            n = 0
            for fx in competitifs + avant_ko:
                dt = _parse_dt(_date_de(fx))
                if dt and seuil <= dt.timestamp() <= ko.timestamp():
                    n += 1
            eq.matchs_14j = n

        # Match suivant (repère un enchaînement européen)
        if ko:
            apres = [fx for fx in prevus
                     if _est_competitif(fx)
                     and (_date_de(fx) or "") > ko.isoformat()]
            apres.sort(key=_date_de)
            if apres:
                nxt = apres[0]
                n_dt = _parse_dt(_date_de(nxt))
                eq.match_suivant = {
                    "date": (_date_de(nxt) or "")[:10],
                    "competition": ((nxt.get("league") or {})
                                    .get("name") or ""),
                    "dans_jours": (round((n_dt - ko).total_seconds()
                                         / 86400.0, 1) if n_dt else None),
                }

    # ─── ABSENTS ────────────────────────────────────

    def _remplir_absents(self, fiche: MatchBrief, fixture: Dict,
                         id_home: Optional[int], id_away: Optional[int]):
        """
        Absents déclarés pour CE match — une seule requête pour les
        deux équipes.

        La source publie environ trois jours à l'avance, et seulement
        sur une partie des rencontres. Un retour vide laisse donc
        `absents_dispo` à False : l'app dira « aucune donnée », jamais
        « effectif au complet ».
        """

        fid = (fixture.get("fixture") or {}).get("id")
        if not fid:
            return

        data = self.api._request("/injuries", {"fixture": fid})
        if not data or data.get("errors"):
            return

        lignes = data.get("response") or []
        if not lignes:
            return

        # L'API sert des doublons (jusqu'à 50 % des lignes selon les
        # cas) : sans déduplication, le compteur d'absents double.
        vus = set()
        for ligne in lignes:
            joueur = ligne.get("player") or {}
            equipe = ligne.get("team") or {}
            cle = (equipe.get("id"), joueur.get("id"))
            if cle in vus or not joueur.get("name"):
                continue
            vus.add(cle)

            absent = Absent(
                nom=joueur.get("name") or "?",
                motif=_fr_motif(joueur.get("reason")),
                certain=(str(joueur.get("type") or "").strip().lower()
                         != "questionable"),
            )
            if equipe.get("id") == id_home:
                fiche.domicile.absents.append(absent)
            elif equipe.get("id") == id_away:
                fiche.exterieur.absents.append(absent)

        # La requête a répondu : la donnée existe pour ce match, même
        # si une équipe n'a personne de listé.
        for eq, ident in ((fiche.domicile, id_home),
                          (fiche.exterieur, id_away)):
            if ident:
                eq.absents_dispo = True

    # ─── SUSPENSIONS PROBABLES ──────────────────────

    def _remplir_rouges(self, eq: FicheEquipe, joues: List[Dict]):
        """
        Carton rouge au dernier match joué → suspension quasi certaine.

        C'est le seul fait réellement PROSPECTIF de cette fiche : il
        est connu jusqu'à huit jours avant le match, là où la liste
        des absents n'arrive que trois jours avant, quand elle arrive.

        Limite honnête : l'API ne donne ni la DURÉE de la suspension
        (un à trois matchs) ni sa portée entre compétitions. On
        signale, on n'affirme pas.
        """

        competitifs = [fx for fx in joues
                       if _est_competitif(fx) and _est_joue(fx)]
        if not competitifs:
            return
        competitifs.sort(key=_date_de, reverse=True)

        fid = (competitifs[0].get("fixture") or {}).get("id")
        if not fid:
            return

        data = self.api._request("/fixtures/events", {"fixture": fid})
        if not data or data.get("errors"):
            return
        eq.rouges_dispo = True

        equipe_ref = _nom_equipe_de(competitifs[0], eq.nom)
        for ev in (data.get("response") or []):
            detail = str(ev.get("detail") or "").lower()
            if "red card" not in detail:
                continue
            nom_eq = ((ev.get("team") or {}).get("name") or "")
            if equipe_ref and nom_eq != equipe_ref:
                continue
            joueur = ((ev.get("player") or {}).get("name") or "?")
            if joueur not in eq.rouges:
                eq.rouges.append(joueur)

    # ─── CLASSEMENT ─────────────────────────────────

    def _remplir_classement(self, fiche: MatchBrief, league: str,
                            id_home: Optional[int],
                            id_away: Optional[int]):
        """
        Rang, points et enjeu.

        GARDE-FOU INDISPENSABLE : hors saison ou avant la première
        journée, l'API renvoie toutes les équipes à zéro point dans
        l'ordre ALPHABÉTIQUE, en gardant le champ « description »
        rempli. Lu sans précaution, cela produit « Auxerre 1er de
        Ligue 1, place qualificative Ligue des Champions » — une
        affirmation fausse, présentée comme une donnée vérifiée.
        """

        lid = LEAGUE_IDS[league]
        saison = self.api._current_season()
        cle = (lid, saison)

        if cle not in self._standings_cache:
            data = self.api._request("/standings",
                                     {"league": lid, "season": saison})
            table = {}
            if data and not data.get("errors"):
                for bloc in (data.get("response") or []):
                    groupes = ((bloc.get("league") or {})
                               .get("standings") or [])
                    for groupe in groupes:
                        for r in groupe:
                            tid = ((r.get("team") or {}).get("id"))
                            if tid:
                                table[tid] = r
            self._standings_cache[cle] = table

        table = self._standings_cache[cle]
        if not table:
            return

        for eq, ident in ((fiche.domicile, id_home),
                          (fiche.exterieur, id_away)):
            r = table.get(ident) if ident else None
            if not r:
                continue
            joues = ((r.get("all") or {}).get("played") or 0)
            if joues <= 0:
                continue          # classement vide : on n'affiche RIEN

            eq.rang = r.get("rank")
            eq.points = r.get("points")
            eq.journees = joues
            eq.classement_fiable = joues >= MIN_JOURNEES_CLASSEMENT
            if eq.classement_fiable:
                eq.enjeu = _fr_enjeu(r.get("description"))

        if (fiche.domicile.journees or fiche.exterieur.journees) and not (
                fiche.domicile.classement_fiable
                or fiche.exterieur.classement_fiable):
            fiche.avertissements.append(
                f"Classement ignoré : moins de "
                f"{MIN_JOURNEES_CLASSEMENT} journées jouées, il ne "
                f"veut encore rien dire")

    # ─── MÉTÉO (Open-Meteo, gratuit, hors quota) ────

    def _meteo(self, fixture: Dict, ko: datetime) -> Optional[Dict]:
        """
        Prévision à l'heure du coup d'envoi.

        Limites assumées : le géocodage porte sur la VILLE, pas sur le
        stade (précision de l'ordre de 5 à 15 km), et l'API ne dit pas
        si l'enceinte est couverte — sur un stade à toit fermé, le
        signal est un faux positif.
        """

        ville = (((fixture.get("fixture") or {}).get("venue") or {})
                 .get("city") or "").strip()
        if not ville:
            return None

        # Plus de 16 jours : hors portée du modèle de prévision
        if (ko - datetime.now(timezone.utc)).days > 15:
            return None

        try:
            g = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": ville, "count": 1, "language": "fr"},
                timeout=12)
            res = (g.json().get("results") or [])
            if not res:
                return None
            lat, lon = res[0]["latitude"], res[0]["longitude"]

            p = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "hourly": ("temperature_2m,precipitation,"
                               "wind_speed_10m"),
                    "start_date": ko.date().isoformat(),
                    "end_date": ko.date().isoformat(),
                    "timezone": "UTC",
                }, timeout=12)
            h = p.json().get("hourly") or {}
            heures = h.get("time") or []
            cible = ko.strftime("%Y-%m-%dT%H:00")
            if cible not in heures:
                return None
            i = heures.index(cible)
            return {
                "ville": ville,
                "temperature": h["temperature_2m"][i],
                "pluie_mm": h["precipitation"][i],
                "vent_kmh": h["wind_speed_10m"][i],
            }
        except Exception:
            return None


# ══════════════════════════════════════════════════════
#  OUTILS
# ══════════════════════════════════════════════════════

def _date_de(fx: Dict) -> Optional[str]:
    return (fx.get("fixture") or {}).get("date")


def _parse_dt(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None


def _est_competitif(fx: Dict) -> bool:
    """Exclut les amicaux : ils faussent le calcul de repos."""
    lid = (fx.get("league") or {}).get("id")
    nom = str((fx.get("league") or {}).get("name") or "").lower()
    if lid in LIGUES_AMICALES:
        return False
    return "friendl" not in nom and "amical" not in nom


def _est_joue(fx: Dict) -> bool:
    statut = ((fx.get("fixture") or {}).get("status") or {}).get("short")
    return statut in STATUTS_JOUES


def _adversaire(fx: Dict, nom_equipe: str) -> str:
    eq = fx.get("teams") or {}
    dom = (eq.get("home") or {}).get("name") or ""
    ext = (eq.get("away") or {}).get("name") or ""
    ref = (nom_equipe or "").lower()
    if ref and ref in dom.lower():
        return ext
    if ref and ref in ext.lower():
        return dom
    return f"{dom} / {ext}"


def _nom_equipe_de(fx: Dict, nom_equipe: str) -> Optional[str]:
    """Nom API de l'équipe dans cette rencontre (pour filtrer les events)."""
    eq = fx.get("teams") or {}
    ref = (nom_equipe or "").lower()
    for cote in ("home", "away"):
        nom = (eq.get(cote) or {}).get("name") or ""
        if ref and ref in nom.lower():
            return nom
    return None


def _score(fx: Dict) -> str:
    buts = fx.get("goals") or {}
    d, e = buts.get("home"), buts.get("away")
    return f"{d}-{e}" if d is not None and e is not None else "—"


def _fr_enjeu(desc: Optional[str]) -> Optional[str]:
    if not desc:
        return None
    d = str(desc).lower()
    if "relegation" in d:
        return "zone de relégation"
    if "champions league" in d:
        return "qualification Ligue des Champions"
    if "europa league" in d:
        return "qualification Ligue Europa"
    if "conference" in d:
        return "qualification Conference League"
    if "play-off" in d or "playoff" in d:
        return "barrages"
    return str(desc)


# ── Instance partagée ──────────────────────────────────

_BUILDER: Optional[MatchBriefBuilder] = None


def get_brief_builder() -> MatchBriefBuilder:
    global _BUILDER
    if _BUILDER is None:
        _BUILDER = MatchBriefBuilder()
    return _BUILDER
