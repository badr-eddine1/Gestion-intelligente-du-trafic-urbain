"""
Agent Q-learning pour contrôle de feux de signalisation
"""

import numpy as np
import random
from collections import defaultdict
from typing import Tuple, Optional
import pickle

class QLearningAgent:
    """
    Agent utilisant l'algorithme Q-learning tabulaire
    """
    
    def __init__(self,
                 alpha: float = 0.1,      # Taux d'apprentissage
                 gamma: float = 0.95,     # Facteur d'actualisation
                 epsilon: float = 1.0,    # Taux d'exploration initial
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Args:
            alpha: taux d'apprentissage (0 < alpha <= 1)
            gamma: facteur d'actualisation (0 <= gamma <= 1)
            epsilon: probabilité d'exploration initiale
            epsilon_min: valeur minimale d'epsilon
            epsilon_decay: facteur de décroissance d'epsilon
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Table Q: dictionnaire {état: [Q(s,a0), Q(s,a1)]}
        self.Q = defaultdict(lambda: [0.0, 0.0])
        
        # Statistiques
        self.episode_rewards = []
        self.epsilon_history = []
        
    def act(self, state: tuple) -> int:
        """
        Choisit une action selon la politique ε-greedy
        
        Args:
            state: l'état courant (clé)
            
        Returns:
            action (0 ou 1)
        """
        if random.random() < self.epsilon:
            # Exploration: action aléatoire
            return random.randint(0, 1)
        else:
            # Exploitation: meilleure action
            q_values = self.Q[state]
            # En cas d'égalité, choisir arbitrairement la première
            return int(np.argmax(q_values))
    
    def learn(self, 
              state: tuple, 
              action: int, 
              reward: float, 
              next_state: tuple) -> float:
        """
        Met à jour la table Q avec l'équation de Bellman
        
        Returns:
            td_error: erreur TD
        """
        # Q-learning update:
        # Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        best_next = max(self.Q[next_state])  # max_a' Q(s',a')
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        
        return td_error
    
    def decay_epsilon(self):
        """Diminue l'exploration au fil du temps"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def get_best_action(self, state: tuple) -> int:
        """Retourne la meilleure action (sans exploration)"""
        return int(np.argmax(self.Q[state]))
    
    def save(self, filepath: str):
        """Sauvegarde la table Q"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.Q), f)
    
    def load(self, filepath: str):
        """Charge une table Q"""
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            self.Q = defaultdict(lambda: [0.0, 0.0], loaded)