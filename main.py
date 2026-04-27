"""
Script principal pour exécuter l'entraînement et l'évaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

from simulation import Intersection
from agent import QLearningAgent
from train import train_agent
from evaluate import compare_policies, plot_comparison
from baseline import FixedTimePolicy, GreedyPolicy


def demonstrate_agent(arrival_rates=None, n_steps=100):
    """
    Démonstration en direct de l'agent
    """
    if arrival_rates is None:
        arrival_rates = {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3}
    
    # Entraîner l'agent
    print("Entraînement de l'agent...")
    env = Intersection(arrival_rates=arrival_rates, max_queue=5)
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.05)  # Faible exploration
    
    train_agent(env, agent, n_episodes=300, steps_per_episode=100, verbose=True)
    
    # Démonstration
    print(f"\n Démonstration sur {n_steps} pas de temps")
    env.reset()
    
    # Pour visualisation
    queue_history = {'N': [], 'S': [], 'E': [], 'O': []}
    phase_history = []
    action_history = []
    
    for step in range(n_steps):
        state = env.get_state_key()
        action = agent.get_best_action(state)
        reward, next_state = env.step(action)
        
        # Enregistrer pour visualisation
        for dir_name in queue_history:
            queue_history[dir_name].append(env.queues[dir_name])
        phase_history.append(env.phase)
        action_history.append(action)
        
        # Affichage simple
        if step % 20 == 0:
            queues = env.get_queue_lengths()
            print(f"Step {step:3d} | Phase: {'NS' if env.phase==0 else 'EO'} | "
                  f"Files: N={queues[0]} S={queues[1]} E={queues[2]} O={queues[3]}")
    
    # Visualisation finale
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Files d'attente
    for dir_name, history in queue_history.items():
        axes[0].plot(history, label=dir_name)
    axes[0].set_xlabel("Pas de temps")
    axes[0].set_ylabel("Longueur de la file")
    axes[0].set_title("Évolution des files d'attente")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Phase
    axes[1].step(range(len(phase_history)), phase_history, where='mid', label='Phase (0=NS, 1=EO)')
    axes[1].set_xlabel("Pas de temps")
    axes[1].set_ylabel("Phase")
    axes[1].set_title("Changements de phase")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("demonstration.png", dpi=150)
    plt.show()


def parameter_study():
    """
    Étude de l'influence des paramètres α et ε
    """
    arrival_rates = {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3}
    
    alphas = [0.05, 0.1, 0.3, 0.5]
    decays = [0.99, 0.995, 0.999]
    
    results = {}
    
    for alpha in alphas:
        for decay in decays:
            print(f"\nTest: α={alpha}, ε_decay={decay}")
            env = Intersection(arrival_rates=arrival_rates, max_queue=5)
            agent = QLearningAgent(alpha=alpha, gamma=0.95, epsilon_decay=decay)
            rewards, _ = train_agent(env, agent, n_episodes=300, verbose=False)
            results[(alpha, decay)] = np.mean(rewards[-50:])
    
    # Affichage des résultats
    print("\n=== Résultats de l'étude des paramètres ===")
    print("(α, ε_decay) → Récompense moyenne (50 derniers épisodes)")
    for (alpha, decay), reward in sorted(results.items(), key=lambda x: -x[1]):
        print(f"α={alpha}, ε_decay={decay}: {reward:.1f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projet Trafic - Agent RL")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'eval', 'demo', 'study', 'all'],
                        help='Mode d\'exécution')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'all']:
        print("\n=== ENTRAÎNEMENT ===")
        from train import train_with_params
        agent, rewards = train_with_params(
            {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3},
            n_episodes=500
        )
    
    if args.mode in ['eval', 'all']:
        print("\n=== ÉVALUATION ===")
        from evaluate import run_scenarios
        run_scenarios()
    
    if args.mode in ['demo', 'all']:
        print("\n=== DÉMONSTRATION ===")
        demonstrate_agent()
    
    if args.mode == 'study':
        print("\n=== ÉTUDE DES PARAMÈTRES ===")
        parameter_study()
    
    if args.mode == 'all':
        print("\n=== TOUS LES TESTS TERMINÉS ===")