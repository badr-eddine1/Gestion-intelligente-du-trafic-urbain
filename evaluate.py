"""
Script d'évaluation — compare Q-Learning avec les baselines
CORRECTION : utilise l'agent déjà entraîné (agent_final.pkl)
             et la clé d'état à 6 éléments (qN,qS,qE,qO,phase,orange)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from simulation import Intersection
from agent import QLearningAgent
from baseline import FixedTimePolicy, GreedyPolicy, RandomPolicy


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER Q-LEARNING — clé d'état corrigée à 6 éléments
# ══════════════════════════════════════════════════════════════════════════════

class QLearningWrapper:
    """
    Adapte l'agent Q-Learning à l'interface act(state) des baselines.
    Reconstruit la clé à 6 éléments depuis l'état (queues_tuple, phase).
    in_orange = 0 pendant l'évaluation (orange désactivé).
    """
    def __init__(self, agent: QLearningAgent):
        self.agent = agent

    def act(self, state: tuple) -> int:
        queues, phase = state
        # Clé à 6 éléments : (qN, qS, qE, qO, phase, in_orange=0)
        state_key = (queues[0], queues[1], queues[2], queues[3], phase, 0)
        return self.agent.get_best_action(state_key)

    def reset(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION D'UNE POLITIQUE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(env: Intersection,
                    policy,
                    n_episodes: int       = 50,
                    steps_per_episode: int = 200) -> Dict[str, List[float]]:
    """
    Évalue une politique sur n_episodes épisodes.

    Returns:
        dict avec rewards, waiting times, moyennes et écarts-types
    """
    all_rewards      = []
    all_waiting_times = []

    for _ in range(n_episodes):
        env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()

        episode_reward  = 0
        episode_waiting = []

        for _ in range(steps_per_episode):
            state  = env.get_state()
            action = policy.act(state)
            reward, _ = env.step(action)
            episode_reward += reward
            episode_waiting.append(env.get_total_waiting())

        all_rewards.append(episode_reward)
        all_waiting_times.append(np.mean(episode_waiting))

    return {
        'rewards':     all_rewards,
        'mean_waiting': all_waiting_times,
        'avg_reward':  np.mean(all_rewards),
        'std_reward':  np.std(all_rewards),
        'avg_waiting': np.mean(all_waiting_times),
        'std_waiting': np.std(all_waiting_times),
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPARAISON DES POLITIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compare_policies(arrival_rates: Dict[str, float],
                     agent_path: str = None,
                     n_episodes: int = 50) -> Dict[str, Dict]:
    """
    Compare Q-Learning et les 3 baselines sur un scénario donné.

    Args:
        arrival_rates : taux d'arrivée par direction
        agent_path    : chemin vers la table Q sauvegardée (agent_final.pkl)
                        → si None, entraîne un nouvel agent (200 épisodes)
        n_episodes    : nombre d'épisodes d'évaluation par politique
    """
    env = Intersection(arrival_rates=arrival_rates, max_queue=5)

    # ── Charger ou entraîner l'agent Q-Learning ──────────────────────────────
    agent = QLearningAgent()

    if agent_path:
        try:
            agent.load(agent_path)
            print(f"Agent chargé depuis {agent_path}")
        except FileNotFoundError:
            print(f"Fichier {agent_path} introuvable → entraînement rapide")
            agent_path = None

    if not agent_path:
        print("Entraînement d'un nouvel agent Q-Learning (200 épisodes)...")
        from train import train_agent
        train_agent(env, agent, n_episodes=200, visualize_q=False, verbose=False)
        print("Entraînement terminé")

    # ── Définir les politiques ───────────────────────────────────────────────
    policies = {
        'Q-learning':       QLearningWrapper(agent),
        'Fixed Time (10s)': FixedTimePolicy(switch_interval=10),
        'Fixed Time (20s)': FixedTimePolicy(switch_interval=20),
        'Greedy (seuil=2)': GreedyPolicy(threshold=2),
        'Random':           RandomPolicy(),
    }

    # ── Évaluation ───────────────────────────────────────────────────────────
    results = {}
    for name, policy in policies.items():
        print(f"Évaluation de {name}...")
        results[name] = evaluate_policy(env, policy, n_episodes=n_episodes)
        print(f"  Reward moyen  : {results[name]['avg_reward']:.1f} "
              f"± {results[name]['std_reward']:.1f}")
        print(f"  Attente moyenne: {results[name]['avg_waiting']:.2f} véhicules")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: Dict[str, Dict], title: str = "") -> None:
    """Graphique de comparaison récompenses + files d'attente."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    policies     = list(results.keys())
    rewards      = [results[p]['avg_reward']  for p in policies]
    reward_stds  = [results[p]['std_reward']  for p in policies]
    waitings     = [results[p]['avg_waiting'] for p in policies]
    waiting_stds = [results[p]['std_waiting'] for p in policies]

    # Graphique récompenses
    bars1 = ax1.bar(policies, rewards, yerr=reward_stds, capsize=5, alpha=0.7)
    ax1.set_ylabel("Récompense moyenne")
    ax1.set_title("Comparaison des récompenses")
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, rewards):
        bar.set_color('green' if val == max(rewards) else
                      'red'   if val == min(rewards) else 'steelblue')

    # Graphique files d'attente
    bars2 = ax2.bar(policies, waitings, yerr=waiting_stds, capsize=5, alpha=0.7)
    ax2.set_ylabel("Nombre moyen de véhicules en attente")
    ax2.set_title("Comparaison des files d'attente")
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, waitings):
        bar.set_color('green' if val == min(waitings) else
                      'red'   if val == max(waitings) else 'steelblue')

    plt.suptitle(f"Comparaison des politiques - {title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/comparison_{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SCÉNARIOS
# ══════════════════════════════════════════════════════════════════════════════

def run_scenarios(agent_path: str = None):
    """
    Lance la comparaison sur 2 scénarios de trafic.

    Args:
        agent_path : chemin vers l'agent déjà entraîné
                     (passé automatiquement par main.py)
    """
    import os
    os.makedirs("plots", exist_ok=True)

    scenarios = {
        'Trafic_equilibre':   {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3},
        'Trafic_asymetrique': {'N': 0.5, 'S': 0.5, 'E': 0.1, 'O': 0.1},
    }

    all_results = {}
    for scenario_name, rates in scenarios.items():
        print(f"\n{'='*50}")
        print(f"Scénario: {scenario_name}")
        print(f"Taux d'arrivée: {rates}")
        print('=' * 50)

        results = compare_policies(rates,
                                   agent_path=agent_path,
                                   n_episodes=20)
        all_results[scenario_name] = results
        plot_comparison(results, title=scenario_name)

    return all_results


if __name__ == "__main__":
    print("=== ÉVALUATION DES POLITIQUES ===")
    import os
    path = "agent_final.pkl" if os.path.exists("agent_final.pkl") else None
    run_scenarios(agent_path=path)
