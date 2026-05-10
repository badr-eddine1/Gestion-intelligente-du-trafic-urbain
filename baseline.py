"""
Politiques de référence (Baselines) pour comparaison avec le Q-Learning

Trois politiques :
  1. FixedTimePolicy  — feu à durée fixe, sans adaptation
  2. GreedyPolicy     — heuristique locale, sert la file la plus chargée
  3. RandomPolicy     — décision aléatoire, borne inférieure absolue
"""

import numpy as np


class FixedTimePolicy:
    """
    Change de phase toutes les N secondes, indépendamment du trafic.
    Simule un feu de signalisation classique.
    """

    def __init__(self, switch_interval: int = 10):
        """
        Args:
            switch_interval : nombre de pas de temps entre chaque changement
        """
        self.switch_interval = switch_interval
        self.counter         = 0
        self.current_phase   = 0

    def act(self, state: tuple) -> int:
        """state = (queues_tuple, phase) — non utilisé ici."""
        self.counter += 1
        if self.counter >= self.switch_interval:
            self.counter = 0
            return 1   # Changer
        return 0       # Maintenir

    def reset(self):
        self.counter = 0


class GreedyPolicy:
    """
    Compare la somme des files NS vs EO et change si l'écart
    dépasse un seuil. Sans apprentissage — heuristique locale.
    """

    def __init__(self, threshold: int = 2):
        """
        Args:
            threshold : écart minimum pour déclencher un changement
        """
        self.threshold = threshold

    def act(self, state: tuple) -> int:
        """state = (queues_tuple, phase)"""
        queues, phase = state
        ns_waiting = queues[0] + queues[1]   # N + S
        eo_waiting = queues[2] + queues[3]   # E + O

        if phase == 0:   # NS vert → changer si EO très chargé
            if eo_waiting > ns_waiting + self.threshold:
                return 1
        else:            # EO vert → changer si NS très chargé
            if ns_waiting > eo_waiting + self.threshold:
                return 1
        return 0

    def reset(self):
        pass


class RandomPolicy:
    """
    Décision uniformément aléatoire.
    Sert de borne inférieure absolue de performance.
    """

    def act(self, state: tuple) -> int:
        return np.random.randint(0, 2)

    def reset(self):
        pass
