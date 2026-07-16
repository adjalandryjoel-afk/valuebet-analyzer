"""
═══════════════════════════════════════════════════════
 MODULE MATCH INTEL — Conseil d'agents d'analyse pré-match
═══════════════════════════════════════════════════════

Un « conseil d'agents » enrichit le modèle statistique avant
un match, puis un chef analyste rédige le verdict :

  • AGENT H2H      : confrontations directes (API-Football)
  • AGENT FORME    : séquence V/N/D récente (API-Football)
  • AGENT CONTEXTE : structure les notes de l'utilisateur (LLM)
  • CHEF ANALYSTE  : synthèse rédigée du dossier chiffré (LLM)

PRINCIPE ANTI-HALLUCINATION (non négociable) :
les agents LLM n'inventent JAMAIS de faits (blessures,
transferts, enjeux...). Le LLM ne fait que (a) structurer le
texte fourni par l'utilisateur, (b) synthétiser des chiffres
qu'on lui donne. Chaque prompt l'exige explicitement.

Les ajustements produits sont des multiplicateurs de lambda
bornés à [0.85, 1.15] : le conseil d'agents nuance le modèle
Poisson/Elo, il ne le remplace jamais. Toute défaillance
(API muette, LLM en erreur, JSON invalide) dégrade proprement
vers « aucun ajustement ».
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from config import APIKeys, OCRConfig
from modules.api_football import get_api_collector
from modules.data_collector import MatchContext


# ══════════════════════════════════════════════════════
#  PARAMÈTRES DES AGENTS
# ══════════════════════════════════════════════════════


def _iso(date_fr: str) -> str:
    """« 17/05/2026 » → « 2026-05-17 » (triable alphabétiquement)."""
    try:
        j, m, a = str(date_fr).split("/")
        return f"{a}-{m}-{j}"
    except ValueError:
        return str(date_fr)


# Bornes absolues des multiplicateurs de lambda
LAMBDA_MULT_MIN = 0.85
LAMBDA_MULT_MAX = 1.15

# Coefficient de l'ajustement H2H — l'historique des
# confrontations est un signal MINEUR, volontairement faible
H2H_TILT_COEF = 0.06

# Taille d'échantillon H2H minimale pour oser un ajustement
H2H_MIN_SAMPLE = 4

# Agent forme : ajustement maximal (±3%) et divergence
# (points/match) entre les 5 derniers et l'ensemble qui
# déclenche l'ajustement
FORM_ADJUST_MAX = 0.03
FORM_DIVERGENCE_THRESHOLD = 0.80
FORM_MIN_MATCHES = 8

# Agent contexte : impact maximal par équipe (±12%)
CONTEXT_IMPACT_MAX = 0.12


# ══════════════════════════════════════════════════════
#  STRUCTURES
# ══════════════════════════════════════════════════════

@dataclass
class IntelReport:
    """Rapport du conseil d'agents pour un match."""

    # Sortie brute de ApiFootballCollector.get_h2h :
    # matches, team1_wins, draws, team2_wins, avg_goals,
    # btts_rate, team1_avg_scored, team2_avg_scored, sample
    h2h: Optional[dict] = None

    # {"sequence": "VNDVV" (plus récent à gauche),
    #  "recent_results": [...]}
    form_home: Optional[dict] = None
    form_away: Optional[dict] = None

    # Multiplicateurs de lambda, bornés à [0.85, 1.15]
    lambda_mult_home: float = 1.0
    lambda_mult_away: float = 1.0

    # Explications françaises des ajustements appliqués
    adjust_reasons: list = field(default_factory=list)

    # Sortie de l'agent contexte (notes utilisateur structurées)
    context_notes_impact: Optional[dict] = None

    # Sources ayant réellement contribué au rapport
    data_sources: list = field(default_factory=list)


# ══════════════════════════════════════════════════════
#  CONSEIL D'AGENTS
# ══════════════════════════════════════════════════════

class MatchIntelligence:
    """
    Orchestre les agents d'analyse pré-match.

    Chaque agent est autonome et défensif : s'il n'a pas ses
    données (pas de clé API, pas d'H2H, pas de notes), il ne
    fait simplement rien — le rapport reste neutre (×1.0).
    """

    def __init__(self):
        self.api = get_api_collector()

        self.openai_client = None
        if APIKeys.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=APIKeys.OPENAI_API_KEY)
            except ImportError:
                print("    ⚠️ Package openai non installé — "
                      "agents LLM désactivés")

    # ─── POINT D'ENTRÉE : ANALYSE ───────────────────

    def analyze(self, home_team: str, away_team: str,
                context: Optional[MatchContext] = None,
                user_notes: str = "") -> IntelReport:
        """
        Réunit le conseil d'agents et retourne un IntelReport.

        Args:
            home_team / away_team : noms officiels des équipes
            context : MatchContext du DataCollector (optionnel,
                      conservé pour enrichissements futurs)
            user_notes : notes libres de l'utilisateur (blessures,
                         enjeux...) — SEULE source de l'agent
                         contexte, jamais complétée par le LLM
        """

        report = IntelReport()

        # Ligue détectée par le TeamMatcher : elle indique à
        # football-data.co.uk où chercher les vraies rencontres
        league = getattr(context, "league", "unknown") or "unknown"

        self._agent_h2h(home_team, away_team, report, league)
        self._agent_form(home_team, is_home=True, report=report,
                         league=league)
        self._agent_form(away_team, is_home=False, report=report,
                         league=league)
        self._agent_context(home_team, away_team, user_notes, report)

        # Borne FINALE des multiplicateurs : quel que soit le
        # cumul des agents, on reste dans [0.85, 1.15]
        report.lambda_mult_home = round(
            self._clamp(report.lambda_mult_home,
                        LAMBDA_MULT_MIN, LAMBDA_MULT_MAX), 4)
        report.lambda_mult_away = round(
            self._clamp(report.lambda_mult_away,
                        LAMBDA_MULT_MIN, LAMBDA_MULT_MAX), 4)

        return report

    # ─── AGENT H2H ──────────────────────────────────

    def _agent_h2h(self, home_team: str, away_team: str,
                   report: IntelReport, league: str = "unknown"):
        """
        Confrontations directes RÉELLES (football-data.co.uk).

        Source principale : les résultats réels des championnats
        européens, gratuits et sans clé. API-Football ne sert plus
        que de repli (son plan gratuit ne couvrait pas la saison en
        cours, et l'abonnement peut expirer).
        """

        h2h = self._h2h_football_data(league, home_team, away_team)
        source = "football-data.co.uk (H2H réel)"

        if h2h is None and hasattr(self.api, "get_h2h"):
            try:
                h2h = self.api.get_h2h(home_team, away_team)
                source = "API-Football (H2H)"
            except Exception:
                h2h = None

        if not isinstance(h2h, dict):
            return

        report.h2h = h2h
        report.data_sources.append(source)

        try:
            sample = int(h2h.get("sample", 0) or 0)
            avg_goals = float(h2h.get("avg_goals", 0) or 0)
            t1_scored = float(h2h.get("team1_avg_scored", 0) or 0)
            t2_scored = float(h2h.get("team2_avg_scored", 0) or 0)
        except (TypeError, ValueError):
            return

        if sample < H2H_MIN_SAMPLE:
            return

        # Effet volontairement FAIBLE : l'H2H est un signal mineur
        tilt = (t1_scored - t2_scored) / max(avg_goals, 1.0)
        mult_home = 1 + H2H_TILT_COEF * tilt
        mult_away = 1 - H2H_TILT_COEF * tilt

        if abs(tilt) < 1e-9:
            return

        report.lambda_mult_home *= mult_home
        report.lambda_mult_away *= mult_away

        dominant = home_team if tilt > 0 else away_team
        report.adjust_reasons.append(
            f"H2H ({sample} confrontations) : {dominant} domine "
            f"historiquement ({t1_scored:.2f} vs {t2_scored:.2f} "
            f"buts/match) → λ domicile ×{mult_home:.3f}, "
            f"λ extérieur ×{mult_away:.3f} (signal mineur)"
        )

    # ─── SOURCE RÉELLE : football-data.co.uk ────────

    @staticmethod
    def _h2h_football_data(league: str, home_team: str,
                           away_team: str) -> Optional[dict]:
        """
        H2H réel, traduit dans la forme attendue par le reste du
        code (mêmes clés que l'ancien collector API-Football, pour
        que synthèse et affichage fonctionnent sans changement).
        """

        try:
            from modules.football_data import get_football_data
            brut = get_football_data().h2h(league, home_team, away_team)
        except Exception:
            return None

        if not isinstance(brut, dict) or not brut.get("total"):
            return None

        return {
            "sample": brut["total"],
            "avg_goals": brut["moy_buts"],
            "team1_avg_scored": brut["buts_home"],
            "team2_avg_scored": brut["buts_away"],
            "team1_wins": brut["victoires_home"],
            "team2_wins": brut["victoires_away"],
            "draws": brut["nuls"],
            "btts_rate": brut["btts_rate"],
            # Clés attendues par l'affichage (_render_intel_section)
            "matches": [
                {"date": m["date"], "home": m["domicile"],
                 "away": m["exterieur"], "score": m["score"]}
                for m in reversed(brut["matchs"])  # plus récent d'abord
            ],
        }

    @staticmethod
    def _form_football_data(league: str, team_name: str) -> Optional[dict]:
        """
        Forme réelle, traduite dans la forme attendue :
        recent_results avec date/adversaire/score/resultat.
        """

        try:
            from modules.football_data import get_football_data
            brut = get_football_data().team_form(league, team_name, n=8)
        except Exception:
            return None

        if not isinstance(brut, dict) or not brut.get("matchs"):
            return None

        return {
            "recent_results": [
                {
                    # Date ISO : _agent_form retrie par ordre
                    # alphabétique décroissant — en jj/mm/aaaa,
                    # « 28/04 » passerait avant « 03/05 »
                    "date": _iso(m["date"]),
                    "adversaire": m["adversaire"],
                    "venue": m["venue"],
                    "score": m["score"],
                    "resultat": m["resultat"],
                }
                for m in reversed(brut["matchs"])  # plus récent d'abord
            ],
        }

    # ─── AGENT FORME ────────────────────────────────

    def _agent_form(self, team_name: str, is_home: bool,
                    report: IntelReport, league: str = "unknown"):
        """
        Séquence V/N/D récente depuis get_team_stats().

        Le champ recent_results est en cours d'ajout côté
        collector : s'il est absent, l'agent ne fait rien.
        Si les 5 derniers matchs divergent fortement de
        l'ensemble (forme brûlante/glaciale), micro-ajustement
        de ±3% maximum.
        """

        # Source principale : résultats réels (football-data.co.uk)
        stats = self._form_football_data(league, team_name)
        source = "football-data.co.uk (forme réelle)"

        if stats is None:
            try:
                stats = self.api.get_team_stats(team_name)
                source = "API-Football (forme)"
            except Exception:
                stats = None

        if not isinstance(stats, dict):
            return

        recent = stats.get("recent_results")
        if not isinstance(recent, (list, tuple)) or not recent:
            return

        results = list(recent)

        # Garantir « plus récent en premier » si des dates existent
        if all(isinstance(r, dict) for r in results) and any(
                r.get("date") for r in results):
            try:
                results.sort(key=lambda r: str(r.get("date") or ""),
                             reverse=True)
            except Exception:
                pass

        letters = [l for l in (self._result_letter(r) for r in results)
                   if l is not None]
        if not letters:
            return

        form = {
            "sequence": "".join(letters[:10]),
            "recent_results": results[:10],
        }
        if is_home:
            report.form_home = form
        else:
            report.form_away = form

        if source not in report.data_sources:
            report.data_sources.append(source)

        # ── Micro-ajustement : les 5 derniers vs l'ensemble ──
        if len(letters) < FORM_MIN_MATCHES:
            return

        points = {"V": 3, "N": 1, "D": 0}
        sample = letters[:15]
        ppm_recent = sum(points[l] for l in sample[:5]) / 5
        ppm_overall = sum(points[l] for l in sample) / len(sample)
        delta = ppm_recent - ppm_overall

        if abs(delta) < FORM_DIVERGENCE_THRESHOLD:
            return

        adjust = self._clamp(delta * 0.025,
                             -FORM_ADJUST_MAX, FORM_ADJUST_MAX)
        if is_home:
            report.lambda_mult_home *= (1 + adjust)
        else:
            report.lambda_mult_away *= (1 + adjust)

        etat = "brûlante" if delta > 0 else "glaciale"
        report.adjust_reasons.append(
            f"Forme {etat} de {team_name} : {ppm_recent:.1f} pt/match "
            f"sur les 5 derniers contre {ppm_overall:.1f} sur "
            f"{len(sample)} matchs → λ ×{1 + adjust:.3f} "
            f"(micro-ajustement ±3% max)"
        )

    @staticmethod
    def _result_letter(entry) -> Optional[str]:
        """
        Convertit une entrée de recent_results en lettre V/N/D
        (Victoire / Nul / Défaite). Accepte plusieurs formats :
        lettre directe, dict avec "result", dict avec buts.
        """

        if isinstance(entry, str):
            letter = entry.strip().upper()[:1]
            return {"V": "V", "W": "V", "N": "N",
                    "D": "D", "L": "D"}.get(letter)

        if isinstance(entry, dict):
            direct = str(entry.get("result") or entry.get("resultat")
                         or "").strip().upper()[:1]
            if direct in ("V", "W"):
                return "V"
            if direct == "N":
                return "N"
            if direct in ("D", "L"):
                return "D"

            # Score texte "2-1" (buts de l'équipe d'abord)
            score = str(entry.get("score") or "")
            if "-" in score:
                try:
                    gf, ga = (int(x) for x in score.split("-", 1))
                    return "V" if gf > ga else ("N" if gf == ga else "D")
                except (TypeError, ValueError):
                    pass

            for k_for, k_against in (("scored", "conceded"),
                                     ("goals_for", "goals_against"),
                                     ("buts_marques", "buts_encaisses")):
                gf, ga = entry.get(k_for), entry.get(k_against)
                if gf is None or ga is None:
                    continue
                try:
                    gf, ga = float(gf), float(ga)
                except (TypeError, ValueError):
                    return None
                if gf > ga:
                    return "V"
                if gf == ga:
                    return "N"
                return "D"

        return None

    # ─── AGENT CONTEXTE (LLM) ───────────────────────

    def _agent_context(self, home_team: str, away_team: str,
                       user_notes: str, report: IntelReport):
        """
        Structure les notes de contexte fournies par l'utilisateur
        via gpt-4o-mini.

        ANTI-HALLUCINATION : le prompt interdit explicitement
        toute connaissance externe — le LLM ne fait que classer
        et pondérer le texte fourni. Sans notes ou sans client
        OpenAI, l'agent ne fait rien. Tout échec LLM → aucun
        ajustement.
        """

        if not user_notes or not user_notes.strip():
            return
        if not self.openai_client:
            return

        prompt = (
            "Tu structures des notes de contexte d'avant-match fournies "
            f"par l'utilisateur pour le match {home_team} vs {away_team}. "
            "N'utilise AUCUNE connaissance externe, uniquement le texte "
            "fourni. N'invente aucun fait : aucune blessure, aucun "
            "transfert, aucun enjeu qui ne figure pas dans les notes. "
            "Réponds en JSON : "
            '{"home_impact": float entre -0.12 et +0.12, '
            '"away_impact": float entre -0.12 et +0.12, '
            '"facteurs": [{"equipe": "home"|"away"|"les deux", '
            '"type": "blessure"|"suspension"|"enjeu"|"fatigue"|'
            '"motivation"|"autre", "resume": str, "impact": str}], '
            '"fiabilite": "haute"|"moyenne"|"basse"} '
            "— home_impact négatif = l'équipe domicile est affaiblie. "
            "Si les notes ne contiennent rien d'exploitable, tout à zéro."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=OCRConfig.VISION_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",
                     "content": f"Notes de l'utilisateur :\n"
                                f"{user_notes.strip()}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=600,
            )

            data = json.loads(response.choices[0].message.content)
            if not isinstance(data, dict):
                return

            home_impact = self._clamp(
                float(data.get("home_impact", 0) or 0),
                -CONTEXT_IMPACT_MAX, CONTEXT_IMPACT_MAX)
            away_impact = self._clamp(
                float(data.get("away_impact", 0) or 0),
                -CONTEXT_IMPACT_MAX, CONTEXT_IMPACT_MAX)

            report.context_notes_impact = data
            report.data_sources.append(
                "Notes utilisateur (structurées par LLM, "
                "aucune connaissance externe)")

            report.lambda_mult_home *= (1 + home_impact)
            report.lambda_mult_away *= (1 + away_impact)

            facteurs = data.get("facteurs")
            if isinstance(facteurs, list):
                for facteur in facteurs:
                    if not isinstance(facteur, dict):
                        continue
                    resume = str(facteur.get("resume", "") or "").strip()
                    if not resume:
                        continue
                    ftype = str(facteur.get("type", "autre") or "autre")
                    equipe = str(facteur.get("equipe", "?") or "?")
                    report.adjust_reasons.append(
                        f"Contexte ({ftype}, {equipe}) : {resume}")

            if home_impact or away_impact:
                report.adjust_reasons.append(
                    f"Impact des notes (fiabilité "
                    f"{data.get('fiabilite', 'inconnue')}) : "
                    f"λ domicile ×{1 + home_impact:.3f}, "
                    f"λ extérieur ×{1 + away_impact:.3f}"
                )

        except Exception as e:
            print(f"      ⚠️ Agent contexte indisponible ({e}) — "
                  "aucun ajustement de contexte")

    # ─── CHEF ANALYSTE (LLM) ────────────────────────

    def synthesize(self, analysis, report: Optional[IntelReport] = None,
                   bankroll: float = 100_000) -> str:
        """
        Rédige le verdict d'avant-match (markdown, français).

        ANTI-HALLUCINATION : le LLM ne reçoit QUE le dossier
        chiffré (probabilités, cotes, value bets, H2H, forme,
        ajustements) et a interdiction d'y ajouter le moindre
        fait extérieur. Retourne "" si pas de client OpenAI ou
        en cas d'échec.

        Args:
            analysis : MatchAnalysis du ValueBetDetector
            report : IntelReport de analyze() (optionnel)
            bankroll : bankroll en FCFA (pour chiffrer les mises)
        """

        if not self.openai_client:
            return ""

        try:
            payload = self._build_payload(analysis, report, bankroll)

            prompt = (
                "Tu es un analyste professionnel de paris sportifs. "
                "À partir EXCLUSIVEMENT des données JSON fournies "
                "(n'ajoute aucun fait extérieur, aucune statistique "
                "non fournie), rédige en français un verdict "
                "d'avant-match de 150-250 mots en markdown : "
                "1) lecture du match en 2-3 phrases, "
                "2) le ou les paris recommandés avec leur logique "
                '(ou "s\'abstenir" si aucun value bet), '
                "3) les risques/réserves. "
                "Ton sobre et professionnel, pas d'emojis, pas de "
                "promesses de gains. Termine par une ligne "
                '"Discipline : ne jamais dépasser la mise conseillée."'
            )

            response = self.openai_client.chat.completions.create(
                model=OCRConfig.VISION_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",
                     "content": json.dumps(payload, ensure_ascii=False,
                                           default=str)},
                ],
                max_tokens=500,
                temperature=0.4,
            )

            return (response.choices[0].message.content or "").strip()

        except Exception as e:
            print(f"      ⚠️ Chef analyste indisponible ({e})")
            return ""

    def _build_payload(self, analysis,
                       report: Optional[IntelReport],
                       bankroll: float) -> Dict:
        """Dossier chiffré compact envoyé au chef analyste."""

        home = getattr(analysis, "home_team", "")
        away = getattr(analysis, "away_team", "")
        odds = getattr(analysis, "odds", {}) or {}
        model_probs = getattr(analysis, "model_probs", {}) or {}

        # Cotes des marchés principaux uniquement (compacité)
        main_keys = ("1", "X", "2", "over_2_5", "under_2_5",
                     "btts_oui", "btts_non")
        cotes = {k: odds[k] for k in main_keys
                 if float(odds.get(k, 0) or 0) > 1}

        # Value bets détectés
        value_bets: List[Dict] = []
        for vb in getattr(analysis, "value_bets", []) or []:
            stake = getattr(vb, "recommended_stake", 0) or 0
            if stake <= 0:
                stake = round(
                    bankroll * (getattr(vb, "kelly_stake", 0) or 0) / 100)
            value_bets.append({
                "marche": getattr(vb, "market", ""),
                "selection": getattr(vb, "selection", ""),
                "cote": getattr(vb, "bookmaker_odds", 0),
                "value_pct": round(
                    (getattr(vb, "value_percentage", 0) or 0) * 100, 1),
                "confiance": getattr(vb, "confidence_score", 0),
                "mise_conseillee_fcfa": stake,
            })

        # Résumé H2H
        h2h_resume = None
        h2h = getattr(report, "h2h", None) if report else None
        if isinstance(h2h, dict):
            h2h_resume = {
                "confrontations": h2h.get("sample"),
                "victoires_domicile": h2h.get("team1_wins"),
                "nuls": h2h.get("draws"),
                "victoires_exterieur": h2h.get("team2_wins"),
                "buts_par_match": h2h.get("avg_goals"),
                "btts_pct": round(
                    float(h2h.get("btts_rate", 0) or 0) * 100),
            }

        def _sequence(form) -> Optional[str]:
            return form.get("sequence") if isinstance(form, dict) else None

        probs_principales = {
            k: model_probs[k] for k in ("1X2", "OU25", "BTTS")
            if k in model_probs
        }

        return {
            "match": {
                "domicile": home,
                "exterieur": away,
                "competition": getattr(analysis, "competition", ""),
            },
            "bankroll_fcfa": bankroll,
            "probabilites_modele": probs_principales,
            "cotes_bookmaker": cotes,
            "marge_bookmaker_pct": getattr(analysis,
                                           "bookmaker_margin", 0),
            "score_le_plus_probable": getattr(analysis,
                                              "predicted_score", ""),
            "value_bets_detectes": value_bets,
            "h2h": h2h_resume,
            "forme_recente": {
                "domicile_plus_recent_a_gauche": _sequence(
                    getattr(report, "form_home", None) if report else None),
                "exterieur_plus_recent_a_gauche": _sequence(
                    getattr(report, "form_away", None) if report else None),
            },
            "ajustements_appliques": (
                list(report.adjust_reasons) if report else []),
            "avertissements": getattr(analysis, "data_warning", "")
                              or "aucun",
        }

    # ─── OUTILS ─────────────────────────────────────

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Borne une valeur dans [lo, hi]."""
        return max(lo, min(hi, value))


# ══════════════════════════════════════════════════════
#  SINGLETON (pour le cache Streamlit côté app)
# ══════════════════════════════════════════════════════

_INSTANCE: Optional[MatchIntelligence] = None


def get_match_intelligence() -> MatchIntelligence:
    """Retourne l'instance partagée de MatchIntelligence."""

    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MatchIntelligence()
    return _INSTANCE
