"""
Module de simulation d'un carrefour à 4 branches
MDP : états, transitions, récompenses
"""

import numpy as np
from typing import Tuple, Dict, List


class Intersection:
    """
    Carrefour avec 4 files d'attente (N, S, E, O)
    Modélise l'environnement MDP pour le Q-Learning
    """

    def __init__(self,
                 arrival_rates: Dict[str, float] = None,
                 max_queue: int = 5,
                 orange_duration: int = 0):
        """
        Args:
            arrival_rates : taux d'arrivée par direction (loi de Poisson)
            max_queue     : capacité maximale d'une file (discrétisation 0..max_queue)
            orange_duration: durée de la phase orange en pas de temps (0 = désactivé)
        """
        if arrival_rates is None:
            arrival_rates = {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3}

        self.arrival_rates   = arrival_rates
        self.max_queue       = max_queue
        self.orange_duration = orange_duration

        # Files d'attente
        self.queues: Dict[str, int] = {'N': 0, 'S': 0, 'E': 0, 'O': 0}

        # Phase actuelle : 0 = NS vert, 1 = EO vert
        self.phase: int         = 0
        self.orange_counter: int = 0
        self.in_orange: bool    = False
        self.time_step: int     = 0

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> Tuple[Tuple[int, ...], int]:
        """Réinitialise le carrefour au début d'un épisode."""
        self.queues        = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        self.phase         = 0
        self.orange_counter = 0
        self.in_orange     = False
        self.time_step     = 0
        return self.get_state()

    # ──────────────────────────────────────────────────────────────────────────
    def get_state(self) -> Tuple[Tuple[int, ...], int]:
        """
        Retourne l'état complet sous forme (queues_tuple, phase).
        Utilisé par les baselines.
        """
        queues_tuple = (self.queues['N'], self.queues['S'],
                        self.queues['E'], self.queues['O'])
        return (queues_tuple, self.phase)

    def get_state_key(self) -> tuple:
        """
        Retourne la clé d'état à 6 éléments utilisée par la table Q.
        (qN, qS, qE, qO, phase, in_orange)
        Taille de l'espace : 6^4 × 2 × 2 = 5 184 états
        """
        return (
            self.queues['N'],
            self.queues['S'],
            self.queues['E'],
            self.queues['O'],
            self.phase,
            int(self.in_orange)   # ← correction du bug : inclus dans la clé
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _generate_arrivals(self) -> None:
        """Génère les arrivées aléatoires selon Poisson(λ) par direction."""
        for direction in self.queues:
            n_arrivals = np.random.poisson(self.arrival_rates[direction])
            self.queues[direction] = min(self.max_queue,
                                         self.queues[direction] + n_arrivals)

    def _process_departures(self) -> None:
        """Fait passer un véhicule par direction active si la file n'est pas vide."""
        if not self.in_orange:
            if self.phase == 0:   # NS vert
                if self.queues['N'] > 0:
                    self.queues['N'] -= 1
                if self.queues['S'] > 0:
                    self.queues['S'] -= 1
            else:                  # EO vert
                if self.queues['E'] > 0:
                    self.queues['E'] -= 1
                if self.queues['O'] > 0:
                    self.queues['O'] -= 1

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: int) -> Tuple[float, tuple]:
        """
        Exécute un pas de temps.

        Args:
            action : 0 = maintenir phase, 1 = changer phase

        Returns:
            (reward, next_state_key)

        Récompense :
            R = −total_waiting − 1 si changement
            • Pénalise l'attente cumulée
            • Pénalise les changements inutiles (−1) pour éviter les oscillations
        """
        self.time_step += 1

        # ── Gestion du changement de phase ────────────────────────────────────
        if action == 1 and not self.in_orange and self.orange_counter == 0:
            if self.orange_duration > 0:
                self.in_orange      = True
                self.orange_counter = self.orange_duration
            else:
                self.phase = 1 - self.phase   # Changement immédiat

        # ── Phase orange ──────────────────────────────────────────────────────
        if self.in_orange:
            self.orange_counter -= 1
            if self.orange_counter == 0:
                self.in_orange = False
                self.phase     = 1 - self.phase

        # ── Écoulement + arrivées ─────────────────────────────────────────────
        self._process_departures()
        self._generate_arrivals()

        # ── Récompense ────────────────────────────────────────────────────────
        total_waiting = sum(self.queues.values())
        reward = -total_waiting
        if action == 1:
            reward -= 1   # Pénalité de changement

        return reward, self.get_state_key()

    # ──────────────────────────────────────────────────────────────────────────
    def get_queue_lengths(self) -> List[int]:
        """Retourne [N, S, E, O]."""
        return [self.queues[d] for d in ['N', 'S', 'E', 'O']]

    def get_total_waiting(self) -> int:
        """Retourne le nombre total de véhicules en attente."""
        return sum(self.queues.values())
