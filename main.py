"""
Script principal — Projet IAD & SMA Partie 1
Gestion intelligente du trafic urbain — Agent Q-Learning

Usage :
    python main.py --mode all    # Tout lancer (entraînement + évaluation + démo)
    python main.py --mode train  # Entraînement uniquement
    python main.py --mode eval   # Évaluation uniquement
    python main.py --mode demo   # Démonstration live
    python main.py --mode study  # Étude des hyperparamètres
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from simulation import Intersection
from agent import QLearningAgent
from train import train_agent, train_with_params, plot_training_curves
from evaluate import compare_policies, plot_comparison, run_scenarios
from baseline import FixedTimePolicy, GreedyPolicy, RandomPolicy


# ══════════════════════════════════════════════════════════════════════════════
# DÉMONSTRATION LIVE
# ══════════════════════════════════════════════════════════════════════════════

def demonstrate_agent(arrival_rates: dict = None, n_steps: int = 100):
    """
    Démonstration en direct de l'agent en mode exploitation.
    Affiche l'état du carrefour toutes les 20 étapes et génère
    le graphique demonstration.png.
    """
    if arrival_rates is None:
        arrival_rates = {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3}

    os.makedirs("plots", exist_ok=True)

    # ── Charger ou entraîner l'agent ─────────────────────────────────────────
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.01)

    if os.path.exists("agent_final.pkl"):
        agent.load("agent_final.pkl")
        print("Agent chargé depuis agent_final.pkl")
    else:
        print("Entraînement de l'agent pour la démonstration (300 épisodes)...")
        env_train = Intersection(arrival_rates=arrival_rates, max_queue=5)
        train_agent(env_train, agent, n_episodes=300,
                    steps_per_episode=100, visualize_q=True, verbose=True)

    # ── Démonstration ─────────────────────────────────────────────────────────
    env = Intersection(arrival_rates=arrival_rates, max_queue=5)
    env.reset()

    queue_history = {'N': [], 'S': [], 'E': [], 'O': []}
    phase_history = []
    total_reward  = 0

    print(f"\n{'='*60}")
    print(f"  DÉMONSTRATION SUR {n_steps} PAS DE TEMPS")
    print(f"{'='*60}")

    for step in range(n_steps):
        state_key = env.get_state_key()
        action    = agent.get_best_action(state_key)
        reward, _ = env.step(action)
        total_reward += reward

        for d in queue_history:
            queue_history[d].append(env.queues[d])
        phase_history.append(env.phase)

        if step % 20 == 0:
            q = env.get_queue_lengths()
            phase_str = 'NS' if env.phase == 0 else 'EO'
            print(f"Step {step:3d} | Phase: {phase_str} | "
                  f"Files: N={q[0]} S={q[1]} E={q[2]} O={q[3]}")

    print(f"\nRécompense totale : {total_reward:.1f}")

    # ── Graphique ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for d, history in queue_history.items():
        axes[0].plot(history, label=d)
    axes[0].set_xlabel("Pas de temps")
    axes[0].set_ylabel("Longueur de la file")
    axes[0].set_title("Évolution des files d'attente")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].step(range(len(phase_history)), phase_history,
                 where='mid', label='Phase (0=NS, 1=EO)')
    axes[1].set_xlabel("Pas de temps")
    axes[1].set_ylabel("Phase")
    axes[1].set_title("Changements de phase")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/demonstration.png", dpi=150)
    plt.show()
    print("Graphique sauvegardé → plots/demonstration.png")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTUDE DES HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

def parameter_study():
    """Étudie l'influence de α et ε_decay sur la convergence."""
    arrival_rates = {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3}
    alphas        = [0.05, 0.1, 0.3, 0.5]
    decays        = [0.99, 0.995, 0.999]
    results       = {}

    for alpha in alphas:
        for decay in decays:
            print(f"\nTest: α={alpha}, ε_decay={decay}")
            env   = Intersection(arrival_rates=arrival_rates, max_queue=5)
            agent = QLearningAgent(alpha=alpha, gamma=0.95,
                                   epsilon_decay=decay, epsilon_min=0.01)
            rewards, _ = train_agent(env, agent, n_episodes=300,
                                     visualize_q=False, verbose=False)
            results[(alpha, decay)] = np.mean(rewards[-50:])

    print("\n" + "=" * 60)
    print("RÉSULTATS ÉTUDE DES PARAMÈTRES")
    print("=" * 60)
    for (a, d), r in sorted(results.items(), key=lambda x: -x[1]):
        print(f"α={a:.2f}, ε_decay={d:.3f} → {r:.1f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projet Trafic — Agent Q-Learning IAD & SMA")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'eval', 'demo', 'study', 'all'],
                        help="Mode d'exécution")
    args = parser.parse_args()

    os.makedirs("plots", exist_ok=True)

    # ── ENTRAÎNEMENT ──────────────────────────────────────────────────────────
    if args.mode in ['train', 'all']:
        print("\n=== ENTRAÎNEMENT ===")
        agent, rewards = train_with_params(
            {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3},
            n_episodes=500
        )
        # Sauvegarder l'agent entraîné pour l'évaluation
        agent.save("agent_final.pkl")

    # ── ÉVALUATION ────────────────────────────────────────────────────────────
    if args.mode in ['eval', 'all']:
        print("\n=== ÉVALUATION ===")
        # Utilise l'agent déjà entraîné si disponible
        agent_path = "agent_final.pkl" if os.path.exists("agent_final.pkl") else None
        run_scenarios(agent_path=agent_path)

    # ── DÉMONSTRATION ─────────────────────────────────────────────────────────
    if args.mode in ['demo', 'all']:
        print("\n=== DÉMONSTRATION ===")
        demonstrate_agent()

    # ── ÉTUDE DES PARAMÈTRES ──────────────────────────────────────────────────
    if args.mode == 'study':
        print("\n=== ÉTUDE DES PARAMÈTRES ===")
        parameter_study()

    if args.mode == 'all':
        print("\n=== TOUS LES TESTS TERMINÉS ===")
