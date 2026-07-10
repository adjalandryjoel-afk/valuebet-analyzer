"""
═══════════════════════════════════════════════════════
 MODULE XGBOOST MODEL — Modèle prédictif Machine
 Learning basé sur XGBoost (Gradient Boosting)
═══════════════════════════════════════════════════════

XGBoost est l'algorithme de référence pour la prédiction
football. Il capture les relations non-linéaires entre
les features et les résultats.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss
)
import xgboost as xgb
import joblib

from config import Paths
from modules.data_collector import MatchContext, TeamStats
from modules.xg_scraper import TeamXGProfile


@dataclass
class XGBoostPrediction:
    """Résultat d'une prédiction XGBoost."""

    prob_home_win: float = 0.0
    prob_draw: float = 0.0
    prob_away_win: float = 0.0

    # Probabilités brutes (avant calibration)
    raw_probs: List[float] = field(default_factory=list)

    # Features utilisées
    features_used: Dict = field(default_factory=dict)
    feature_importance: Dict = field(default_factory=dict)

    confidence: float = 0.0
    model_name: str = "XGBoost"


class XGBoostPredictor:
    """
    Modèle prédictif basé sur XGBoost pour la classification
    des résultats de matchs de football (1X2).

    Features utilisées :
    - Ratings Elo des deux équipes
    - xG moyens (pour et contre, domicile et extérieur)
    - Forme récente (points par match sur les 5 derniers)
    - Différence de buts marqués/encaissés
    - Head-to-Head historique
    - Position au classement
    - Avantage domicile
    - Jours de repos
    """

    MODEL_PATH = os.path.join("models", "xgboost_1x2.pkl")
    SCALER_PATH = os.path.join("models", "scaler_1x2.pkl")

    # Noms des features
    FEATURE_NAMES = [
        # Elo
        "home_elo",
        "away_elo",
        "elo_diff",

        # xG
        "home_xg_for",
        "home_xg_against",
        "away_xg_for",
        "away_xg_against",
        "xg_diff",

        # xG récents (5 derniers matchs)
        "home_recent_xg_for",
        "home_recent_xg_against",
        "away_recent_xg_for",
        "away_recent_xg_against",

        # Forme récente
        "home_form_score",
        "away_form_score",
        "form_diff",

        # Buts
        "home_avg_goals_scored",
        "home_avg_goals_conceded",
        "away_avg_goals_scored",
        "away_avg_goals_conceded",
        "home_goal_diff",
        "away_goal_diff",

        # Domicile/Extérieur spécifique
        "home_avg_scored_at_home",
        "home_avg_conceded_at_home",
        "away_avg_scored_away",
        "away_avg_conceded_away",

        # Classement
        "home_position",
        "away_position",
        "position_diff",

        # Points
        "home_ppg",
        "away_ppg",
        "ppg_diff",

        # Surperformance xG
        "home_overperformance",  # buts réels - xG
        "away_overperformance",

        # Moyenne de buts de la ligue
        "league_avg_goals",
    ]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._load_model()

    # ─── CHARGEMENT / SAUVEGARDE DU MODÈLE ──────────

    def _load_model(self):
        """Charge un modèle pré-entraîné s'il existe."""

        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = joblib.load(self.MODEL_PATH)
                self.scaler = joblib.load(self.SCALER_PATH)
                self.is_trained = True
                print("    ✅ Modèle XGBoost chargé")
            except Exception as e:
                print(f"    ⚠️ Erreur chargement modèle: {e}")
                self._create_default_model()
        else:
            self._create_default_model()

    def _create_default_model(self):
        """Crée un modèle XGBoost avec les hyperparamètres optimaux."""

        self.model = xgb.XGBClassifier(
            # Architecture
            n_estimators=500,
            max_depth=6,
            min_child_weight=3,

            # Régularisation (éviter l'overfitting)
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,        # L1 regularization
            reg_lambda=1.0,       # L2 regularization
            gamma=0.1,            # Min loss reduction

            # Objectif
            objective='multi:softprob',
            num_class=3,          # 0=Home, 1=Draw, 2=Away
            eval_metric='mlogloss',

            # Divers
            random_state=42,
            n_jobs=-1,
            verbosity=0,

            # Early stopping sera géré pendant le training
            early_stopping_rounds=50,
        )

        self.is_trained = False

    def save_model(self):
        """Sauvegarde le modèle entraîné."""

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)
        print(f"    💾 Modèle sauvegardé : {self.MODEL_PATH}")

    # ─── EXTRACTION DES FEATURES ────────────────────

    def extract_features(self, context: MatchContext,
                         home_elo: float = 1500,
                         away_elo: float = 1500,
                         home_xg_profile: Optional[TeamXGProfile] = None,
                         away_xg_profile: Optional[TeamXGProfile] = None
                         ) -> np.ndarray:
        """
        Extrait le vecteur de features d'un match.

        Transforme toutes les données collectées en un vecteur numérique
        utilisable par XGBoost.
        """

        home = context.home_stats
        away = context.away_stats

        features = {}

        # ── ELO ──
        features["home_elo"] = home_elo
        features["away_elo"] = away_elo
        features["elo_diff"] = home_elo - away_elo

        # ── xG ──
        if home_xg_profile and home_xg_profile.data_available:
            features["home_xg_for"] = home_xg_profile.avg_xg_for
            features["home_xg_against"] = home_xg_profile.avg_xg_against
            features["home_recent_xg_for"] = home_xg_profile.recent_avg_xg_for
            features["home_recent_xg_against"] = home_xg_profile.recent_avg_xg_against
            features["home_overperformance"] = home_xg_profile.goals_minus_xg
        elif home:
            features["home_xg_for"] = home.xg_scored if home.xg_available else home.avg_goals_scored
            features["home_xg_against"] = home.xg_conceded if home.xg_available else home.avg_goals_conceded
            features["home_recent_xg_for"] = features["home_xg_for"]
            features["home_recent_xg_against"] = features["home_xg_against"]
            features["home_overperformance"] = 0.0
        else:
            features["home_xg_for"] = 1.3
            features["home_xg_against"] = 1.2
            features["home_recent_xg_for"] = 1.3
            features["home_recent_xg_against"] = 1.2
            features["home_overperformance"] = 0.0

        if away_xg_profile and away_xg_profile.data_available:
            features["away_xg_for"] = away_xg_profile.avg_xg_for
            features["away_xg_against"] = away_xg_profile.avg_xg_against
            features["away_recent_xg_for"] = away_xg_profile.recent_avg_xg_for
            features["away_recent_xg_against"] = away_xg_profile.recent_avg_xg_against
            features["away_overperformance"] = away_xg_profile.goals_minus_xg
        elif away:
            features["away_xg_for"] = away.xg_scored if away.xg_available else away.avg_goals_scored
            features["away_xg_against"] = away.xg_conceded if away.xg_available else away.avg_goals_conceded
            features["away_recent_xg_for"] = features["away_xg_for"]
            features["away_recent_xg_against"] = features["away_xg_against"]
            features["away_overperformance"] = 0.0
        else:
            features["away_xg_for"] = 1.1
            features["away_xg_against"] = 1.3
            features["away_recent_xg_for"] = 1.1
            features["away_recent_xg_against"] = 1.3
            features["away_overperformance"] = 0.0

        features["xg_diff"] = features["home_xg_for"] - features["away_xg_for"]

        # ── FORME RÉCENTE ──
        features["home_form_score"] = home.recent_form_score if home else 1.5
        features["away_form_score"] = away.recent_form_score if away else 1.5
        features["form_diff"] = features["home_form_score"] - features["away_form_score"]

        # ── BUTS ──
        features["home_avg_goals_scored"] = home.avg_goals_scored if home else 1.3
        features["home_avg_goals_conceded"] = home.avg_goals_conceded if home else 1.2
        features["away_avg_goals_scored"] = away.avg_goals_scored if away else 1.1
        features["away_avg_goals_conceded"] = away.avg_goals_conceded if away else 1.3

        features["home_goal_diff"] = (
            features["home_avg_goals_scored"] - features["home_avg_goals_conceded"]
        )
        features["away_goal_diff"] = (
            features["away_avg_goals_scored"] - features["away_avg_goals_conceded"]
        )

        # ── DOMICILE / EXTÉRIEUR ──
        features["home_avg_scored_at_home"] = home.avg_goals_scored_home if home else 1.4
        features["home_avg_conceded_at_home"] = home.avg_goals_conceded_home if home else 1.1
        features["away_avg_scored_away"] = away.avg_goals_scored_away if away else 1.0
        features["away_avg_conceded_away"] = away.avg_goals_conceded_away if away else 1.4

        # ── CLASSEMENT ──
        features["home_position"] = home.league_position if home else 10
        features["away_position"] = away.league_position if away else 10
        features["position_diff"] = features["away_position"] - features["home_position"]

        # ── POINTS ──
        features["home_ppg"] = home.points_per_game if home else 1.5
        features["away_ppg"] = away.points_per_game if away else 1.5
        features["ppg_diff"] = features["home_ppg"] - features["away_ppg"]

        # ── LIGUE ──
        features["league_avg_goals"] = context.league_avg_goals

        # Convertir en vecteur ordonné
        feature_vector = [features.get(f, 0.0) for f in self.FEATURE_NAMES]

        return np.array(feature_vector).reshape(1, -1), features

    # ─── ENTRAÎNEMENT ────────────────────────────────

    def train(self, training_data: pd.DataFrame):
        """
        Entraîne le modèle XGBoost sur des données historiques.

        Args:
            training_data: DataFrame avec les features et la colonne 'result'
                           result = 0 (Home Win), 1 (Draw), 2 (Away Win)
        """

        print("\n  🎓 Entraînement du modèle XGBoost...")

        # Séparer features et labels
        X = training_data[self.FEATURE_NAMES].values
        y = training_data['result'].values

        # Normaliser
        X_scaled = self.scaler.fit_transform(X)

        # Split temporel (pas de data leakage)
        tss = TimeSeriesSplit(n_splits=5)

        # Cross-validation
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tss.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            val_preds = self.model.predict(X_val)
            fold_acc = accuracy_score(y_val, val_preds)
            cv_scores.append(fold_acc)

            print(f"    Fold {fold+1}: Accuracy = {fold_acc:.3f}")

        print(f"    📊 CV Accuracy moyenne : {np.mean(cv_scores):.3f} "
              f"(±{np.std(cv_scores):.3f})")

        # Entraîner sur toutes les données
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            verbose=False
        )

        self.is_trained = True
        self.save_model()

        # Feature importance
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        print("\n    📈 Top 10 features les plus importantes :")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"      {i+1}. {self.FEATURE_NAMES[idx]}: "
                  f"{importance[idx]:.4f}")

    # ─── PRÉDICTION ──────────────────────────────────

    def predict(self, context: MatchContext,
                home_elo: float = 1500,
                away_elo: float = 1500,
                home_xg_profile: Optional[TeamXGProfile] = None,
                away_xg_profile: Optional[TeamXGProfile] = None
                ) -> XGBoostPrediction:
        """
        Prédit les probabilités 1X2 d'un match.
        """

        # Extraire les features
        feature_vector, features_dict = self.extract_features(
            context, home_elo, away_elo,
            home_xg_profile, away_xg_profile
        )

        prediction = XGBoostPrediction(
            features_used=features_dict
        )

        if not self.is_trained:
            # Fallback : utiliser une estimation simple
            prediction.prob_home_win = 0.45
            prediction.prob_draw = 0.25
            prediction.prob_away_win = 0.30
            prediction.confidence = 30.0
            return prediction

        # Normaliser
        feature_scaled = self.scaler.transform(feature_vector)

        # Prédire les probabilités
        probs = self.model.predict_proba(feature_scaled)[0]

        prediction.prob_home_win = float(probs[0])
        prediction.prob_draw = float(probs[1])
        prediction.prob_away_win = float(probs[2])
        prediction.raw_probs = probs.tolist()

        # Feature importance pour ce match
        if hasattr(self.model, 'feature_importances_'):
            for i, name in enumerate(self.FEATURE_NAMES):
                prediction.feature_importance[name] = float(
                    self.model.feature_importances_[i]
                )

        # Confiance basée sur l'écart entre les probabilités
        max_prob = max(probs)
        prediction.confidence = min(max_prob * 100 + 20, 95)

        return prediction

    # ─── GÉNÉRATION DE DONNÉES D'ENTRAÎNEMENT ───────

    @staticmethod
    def generate_training_data_from_api(seasons: List[int],
                                         leagues: List[str]) -> pd.DataFrame:
        """
        Génère un DataFrame de données d'entraînement
        à partir des données historiques des APIs.

        À implémenter avec les données réelles.
        """

        # Placeholder - à remplir avec les vraies données
        print("    ⚠️ Génération de données d'entraînement requiert des données historiques")
        print("    💡 Utilisez le module backtester pour collecter les données")

        return pd.DataFrame()
