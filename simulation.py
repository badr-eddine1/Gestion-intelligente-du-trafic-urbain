"""
Module de simulation d'un carrefour à 4 branches
"""

import numpy as np
from typing import Tuple, Dict, List

class Intersection:
    """
    Carrefour avec 4 files d'attente (N, S, E, O)
    """
    
    def __init__(self, 
                 arrival_rates: Dict[str, float] = None,
                 max_queue: int = 5,
                 orange_duration: int = 0):
        """
        Args:
            arrival_rates: taux d'arrivée par direction (Poisson)
            max_queue: capacité maximale d'une file (discrétisation)
            orange_duration: durée de la phase orange (0 = pas d'orange)
        """
        if arrival_rates is None:
            # Trafic équilibré par défaut
            arrival_rates = {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3}
        
        self.arrival_rates = arrival_rates
        self.max_queue = max_queue
        self.orange_duration = orange_duration
        
        # Files d'attente
        self.queues: Dict[str, int] = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        
        # Phase actuelle: 0 = NS vert, 1 = EO vert
        self.phase: int = 0
        # Compteur pour l'orange
        self.orange_counter: int = 0
        self.in_orange: bool = False
        
        # Compteur de temps
        self.time_step: int = 0
        
    def reset(self) -> Tuple[Tuple[int, ...], int]:
        """Réinitialise le carrefour"""
        self.queues = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        self.phase = 0
        self.orange_counter = 0
        self.in_orange = False
        self.time_step = 0
        return self.get_state()
    
    def get_state(self) -> Tuple[Tuple[int, ...], int]:
        """
        Retourne l'état complet du système
        Returns:
            (queues, phase): queues = (N, S, E, O)
        """
        queues_tuple = (self.queues['N'], self.queues['S'], 
                       self.queues['E'], self.queues['O'])
        return (queues_tuple, self.phase)
    
    def get_state_key(self) -> tuple:
        """Retourne une clé unique pour l'état (utilisée par Q-learning)"""
        return (self.queues['N'], self.queues['S'],
                self.queues['E'], self.queues['O'], self.phase,
                int(self.in_orange))
    
    def _generate_arrivals(self) -> None:
        """Génère les arrivées aléatoires selon Poisson"""
        for direction in self.queues:
            # Nombre d'arrivées = Poisson(λ)
            n_arrivals = np.random.poisson(self.arrival_rates[direction])
            self.queues[direction] = min(self.max_queue, 
                                         self.queues[direction] + n_arrivals)
    
    def _process_departures(self) -> None:
        """Fait passer les véhicules selon la phase actuelle"""
        if not self.in_orange:
            if self.phase == 0:  # NS vert
                if self.queues['N'] > 0:
                    self.queues['N'] -= 1
                if self.queues['S'] > 0:
                    self.queues['S'] -= 1
            else:  # EO vert
                if self.queues['E'] > 0:
                    self.queues['E'] -= 1
                if self.queues['O'] > 0:
                    self.queues['O'] -= 1
    
    def step(self, action: int) -> Tuple[float, tuple]:
        """
        Exécute un pas de temps
        
        Args:
            action: 0 = maintenir phase, 1 = changer phase
            
        Returns:
            (reward, next_state_key)
        """
        self.time_step += 1
        
        # Gestion du changement de phase avec orange
        if action == 1 and not self.in_orange and self.orange_counter == 0:
            # Initier un changement
            if self.orange_duration > 0:
                self.in_orange = True
                self.orange_counter = self.orange_duration
            else:
                # Changement immédiat
                self.phase = 1 - self.phase
        
        # Gestion de la phase orange
        if self.in_orange:
            self.orange_counter -= 1
            if self.orange_counter == 0:
                self.in_orange = False
                self.phase = 1 - self.phase  # Changement effectif
        
        # Faire passer les véhicules
        self._process_departures()
        
        # Générer les nouvelles arrivées
        self._generate_arrivals()
        
        # Calculer la récompense (négative = pénalité)
        # On pénalise le nombre total de véhicules en attente
        total_waiting = sum(self.queues.values())
        reward = -total_waiting
        
        # Pénalité pour changement de phase trop fréquent
        # Évite que l'agent change à chaque pas sans raison
        if action == 1:
            reward -= 1
        
        return reward, self.get_state_key()
    
    def get_queue_lengths(self) -> List[int]:
        """Retourne les longueurs des files"""
        return [self.queues[d] for d in ['N', 'S', 'E', 'O']]
    
    def get_total_waiting(self) -> int:
        """Retourne le nombre total de véhicules en attente"""
        return sum(self.queues.values())