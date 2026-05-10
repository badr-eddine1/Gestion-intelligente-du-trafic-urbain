"""
Agent Q-Learning tabulaire pour le contrôle des feux de signalisation
Implémenté from scratch — aucune bibliothèque RL externe
"""

import numpy as np
import random
from collections import defaultdict
from typing import Tuple, List
import pickle


class QLearningAgent:
    """
    Agent utilisant l'algorithme Q-Learning tabulaire.

    La table Q est stockée dans un defaultdict :
        Q[état] = [Q(s, maintenir), Q(s, changer)]

    Mise à jour de Bellman :
        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
    """

    def __init__(self,
                 alpha: float         = 0.1,    # Taux d'apprentissage
                 gamma: float         = 0.95,   # Facteur d'actualisation
                 epsilon: float       = 1.0,    # Exploration initiale
                 epsilon_min: float   = 0.01,   # Exploration minimale
                 epsilon_decay: float = 0.995): # Décroissance par épisode
        """
        Justification des paramètres par défaut :
          alpha = 0.1   → convergence stable, évite les oscillations
          gamma = 0.95  → horizon court/moyen, récompenses prochaines valorisées
          epsilon = 1.0 → exploration totale au départ
          epsilon_min = 0.01 → 1 % d'exploration résiduelle (états rares)
          epsilon_decay = 0.995 → transition douce sur ~500 épisodes
        """
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Table Q initialisée à 0 pour tout état non visité
        self.Q = defaultdict(lambda: [0.0, 0.0])

        # Historiques pour visualisation
        self.episode_rewards  = []
        self.epsilon_history  = []

    # ──────────────────────────────────────────────────────────────────────────
    def act(self, state: tuple) -> int:
        """
        Politique ε-greedy :
          - avec proba ε  → action aléatoire (exploration)
          - avec proba 1-ε → meilleure action connue (exploitation)
        """
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.Q[state]))

    def get_best_action(self, state: tuple) -> int:
        """Retourne la meilleure action sans exploration (évaluation)."""
        return int(np.argmax(self.Q[state]))

    # ──────────────────────────────────────────────────────────────────────────
    def learn(self,
              state: tuple,
              action: int,
              reward: float,
              next_state: tuple) -> float:
        """
        Met à jour la table Q avec l'équation de Bellman.

        Returns:
            td_error : erreur de différence temporelle
        """
        best_next  = max(self.Q[next_state])
        td_target  = reward + self.gamma * best_next
        td_error   = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        """Diminue ε après chaque épisode."""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    # ──────────────────────────────────────────────────────────────────────────
    def save(self, filepath: str):
        """Sauvegarde la table Q dans un fichier pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.Q), f)
        print(f"  Agent sauvegardé → {filepath}")

    def load(self, filepath: str):
        """Charge une table Q depuis un fichier pickle."""
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        self.Q = defaultdict(lambda: [0.0, 0.0], loaded)
        print(f"  Agent chargé ← {filepath}  ({len(self.Q)} états)")
