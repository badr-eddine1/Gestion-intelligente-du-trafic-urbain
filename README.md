# Gestion intelligente du trafic urbain — Agent Q-Learning

**Projet IAD & SMA — Partie 1 | Cycle Ingénieur 2025–2026**  
AIT BENIJJA Badr-Eddine · DERBEZ Mehdi · CHAKIR Ilyas

---

## Description

Agent Q-Learning tabulaire contrôlant les feux d'un carrefour à 4 branches (N, S, E, O).  
Les arrivées de véhicules suivent un processus de Poisson. L'agent apprend à minimiser  
le temps d'attente total par essais et erreurs sur 500 épisodes d'entraînement.

---

## Installation

```bash
pip install numpy matplotlib seaborn
```

Python 3.x requis. **Aucune bibliothèque RL externe** — Q-Learning implémenté from scratch.

---

## Structure du projet

```
projetAgent/
├── simulation.py   # Environnement MDP (carrefour, files, récompense)
├── agent.py        # Agent Q-Learning tabulaire (table Q, Bellman, ε-greedy)
├── baseline.py     # Politiques de référence (FixedTime, Greedy, Random)
├── train.py        # Boucle d'entraînement + visualisation table Q
├── evaluate.py     # Évaluation et comparaison des politiques
├── main.py         # Script principal (point d'entrée unique)
├── README.md       # Ce fichier
└── plots/          # Courbes générées automatiquement
```

---

## Exécution

### Tout lancer (recommandé pour la soutenance)
```bash
python main.py --mode all
```

### Entraînement uniquement
```bash
python main.py --mode train
```

### Évaluation et comparaison des baselines
```bash
python main.py --mode eval
```

### Démonstration live de l'agent
```bash
python main.py --mode demo
```

### Étude des hyperparamètres (α, ε_decay)
```bash
python main.py --mode study
```

---

## Paramètres du MDP

| Composant | Valeur | Justification |
|-----------|--------|---------------|
| Espace d'états | 6⁴ × 2 × 2 = **5 184 états** | (qN,qS,qE,qO,phase,orange) |
| Espace d'actions | **{0, 1}** | 0 = maintenir, 1 = changer |
| Récompense R | **−total_waiting − 1** si changement | Pénalise attente + oscillations |
| Facteur γ | **0,95** | Horizon court/moyen terme |
| α (learning rate) | **0,10** | Convergence stable |
| ε initial | **1,00** | Exploration totale au départ |
| ε minimum | **0,01** | Exploration résiduelle |
| ε decay | **0,995** | Transition douce sur 500 épisodes |
| Épisodes | **500** | Convergence observée avant 400 |

---

## Fichiers générés dans plots/

| Fichier | Contenu |
|---------|---------|
| `training_curves.png` | Courbe de récompense + décroissance ε |
| `q_table_ep0100.png` ... `ep0500.png` | Table Q tous les 100 épisodes |
| `q_heatmap_ep0100.png` ... | Heatmap Q(changer)−Q(maintenir) |
| `comparison_Trafic_equilibre.png` | Comparaison politiques — trafic λ=0.3 |
| `comparison_Trafic_asymetrique.png` | Comparaison — trafic asymétrique |
| `demonstration.png` | Évolution files d'attente en live |

---

## Corrections appliquées

1. **Bug clé d'état** — `get_state_key()` inclut maintenant `in_orange` (6 éléments)
2. **Pénalité changement** — `reward -= 1` si action = changer
3. **Évaluation** — utilise `agent_final.pkl` (500 épisodes) au lieu de réentraîner
4. **Wrapper évaluation** — clé d'état corrigée à 6 éléments dans `evaluate.py`
