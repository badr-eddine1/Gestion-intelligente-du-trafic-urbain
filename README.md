# Gestion intelligente du trafic urbain — Agent Q-Learning

Projet IAD & SMA — Partie 1 | Cycle Ingénieur 2025–2026

## Description

Agent Q-Learning tabulaire contrôlant les feux d'un carrefour à 4 branches (N, S, E, O).  
Les arrivées de véhicules suivent un processus de Poisson. L'agent apprend à minimiser le temps d'attente total.

## Installation

```bash
pip install numpy matplotlib seaborn
```

Python 3.x requis. Aucune bibliothèque RL externe (Q-Learning implémenté from scratch).

## Structure du projet

```
projetAgent/
├── simulation.py   # Environnement du carrefour (MDP)
├── agent.py        # Agent Q-Learning tabulaire
├── baseline.py     # Politiques de référence (Fixed Time, Greedy, Random)
├── train.py        # Entraînement + visualisation table Q
├── evaluate.py     # Évaluation et comparaison des politiques
├── main.py         # Script principal
└── plots/          # Courbes exportées (PNG/PDF)
```

## Exécution

### Tout lancer (entraînement + évaluation + démo)
```bash
python main.py --mode all
```

### Entraînement seul
```bash
python main.py --mode train
```

### Évaluation et comparaison avec les baselines
```bash
python main.py --mode eval
```

### Démonstration live de l'agent
```bash
python main.py --mode demo
```

### Étude des paramètres (α, ε)
```bash
python main.py --mode study
```

## Paramètres du MDP

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| α (taux d'apprentissage) | 0.1 | Convergence stable, évite les oscillations |
| γ (facteur d'actualisation) | 0.95 | Horizon temporel court, le trafic évolue vite |
| ε initial | 1.0 | Exploration maximale au début |
| ε_min | 0.01 | Conserve 1% d'exploration résiduelle |
| ε_decay | 0.995 | Transition progressive exploration → exploitation |
| max_queue | 5 | Discrétisation à 6 niveaux (0–5 véhicules) |

## Scénarios de trafic

- **Trafic équilibré** : λ = 0.3 pour toutes les directions
- **Trafic asymétrique** : λ_NS = 0.5, λ_EO = 0.1
