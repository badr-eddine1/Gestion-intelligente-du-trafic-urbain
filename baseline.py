"""
Politiques de référence (baselines) pour comparaison
"""

import numpy as np
from typing import Tuple

class FixedTimePolicy:
    """
    Politique à durée fixe: change de phase périodiquement
    """
    
    def __init__(self, switch_interval: int = 10):
        """
        Args:
            switch_interval: nombre de pas entre chaque changement
        """
        self.switch_interval = switch_interval
        self.counter = 0
        self.current_phase = 0
        
    def act(self, state: tuple) -> int:
        """
        Décide de changer ou non
        
        Args:
            state: (queues, phase) - non utilisé pour cette politique
        """
        self.counter += 1
        if self.counter >= self.switch_interval:
            self.counter = 0
            return 1  # Changer
        return 0  # Maintenir
    
    def reset(self):
        """Réinitialise le compteur"""
        self.counter = 0


class GreedyPolicy:
    """
    Politique gloutonne: change si la file la plus longue est dans l'autre direction
    """
    
    def __init__(self, threshold: int = 2):
        """
        Args:
            threshold: seuil de différence pour déclencher un changement
        """
        self.threshold = threshold
        
    def act(self, state: tuple) -> int:
        """
        state: (queues, phase)
        queues = (N, S, E, O)
        phase: 0 = NS vert, 1 = EO vert
        """
        queues, phase = state
        
        # Somme des files dans chaque direction
        ns_waiting = queues[0] + queues[1]  # N + S
        eo_waiting = queues[2] + queues[3]  # E + O
        
        if phase == 0:  # NS vert actuellement
            # Si EO a beaucoup plus d'attente, changer
            if eo_waiting > ns_waiting + self.threshold:
                return 1
        else:  # EO vert
            if ns_waiting > eo_waiting + self.threshold:
                return 1
        return 0
    
    def reset(self):
        """Réinitialisation (rien à faire)"""
        pass


class RandomPolicy:
    """
    Politique aléatoire (baseline minimale)
    """
    
    def act(self, state: tuple) -> int:
        return np.random.randint(0, 2)
    
    def reset(self):
        pass