"""
Script d'évaluation comparant Q-learning avec les baselines
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from simulation import Intersection
from agent import QLearningAgent
from baseline import FixedTimePolicy, GreedyPolicy, RandomPolicy


def evaluate_policy(env: Intersection,
                    policy,
                    n_episodes: int = 50,
                    steps_per_episode: int = 200) -> Dict[str, List[float]]:
    """
    Évalue une politique sur plusieurs épisodes
    """
    all_rewards = []
    all_waiting_times = []
    
    for episode in range(n_episodes):
        env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()
            
        episode_reward = 0
        episode_waiting = []
        
        for step in range(steps_per_episode):
            state = env.get_state()
            action = policy.act(state)
            reward, _ = env.step(action)
            
            episode_reward += reward
            episode_waiting.append(env.get_total_waiting())
        
        all_rewards.append(episode_reward)
        all_waiting_times.append(np.mean(episode_waiting))
    
    return {
        'rewards': all_rewards,
        'mean_waiting': all_waiting_times,
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'avg_waiting': np.mean(all_waiting_times),
        'std_waiting': np.std(all_waiting_times)
    }


def compare_policies(arrival_rates: Dict[str, float],
                     agent_path: str = None,
                     n_episodes: int = 50) -> Dict[str, Dict]:
    """
    Compare Q-learning avec les baselines
    """
    env = Intersection(arrival_rates=arrival_rates, max_queue=5)
    
    # Wrapper pour Q-learning
    class QLearningWrapper:
        def __init__(self, agent):
            self.agent = agent
        def act(self, state):
            # state = (queues, phase)
            queues, phase = state
            state_key = (queues[0], queues[1], queues[2], queues[3], phase)
            return self.agent.get_best_action(state_key)
        def reset(self):
            pass
    
    # Charger ou créer l'agent Q-learning
    if agent_path and agent_path != "":
        agent = QLearningAgent()
        agent.load(agent_path)
        print("Agent chargé depuis", agent_path)
    else:
        print("Entraînement d'un nouvel agent Q-learning...")
        agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1)
        from train import train_agent
        train_agent(env, agent, n_episodes=200, verbose=False)
        print("Entraînement terminé")
    
    # Définir les politiques
    policies = {
        'Q-learning': QLearningWrapper(agent),
        'Fixed Time (10s)': FixedTimePolicy(switch_interval=10),
        'Fixed Time (20s)': FixedTimePolicy(switch_interval=20),
        'Greedy (seuil=2)': GreedyPolicy(threshold=2),
        'Random': RandomPolicy()
    }
    
    results = {}
    for name, policy in policies.items():
        print(f"Évaluation de {name}...")
        results[name] = evaluate_policy(env, policy, n_episodes=n_episodes)
        print(f"  Reward moyen: {results[name]['avg_reward']:.1f} ± {results[name]['std_reward']:.1f}")
        print(f"  Attente moyenne: {results[name]['avg_waiting']:.2f} véhicules")
    
    return results


def plot_comparison(results: Dict[str, Dict], title: str = ""):
    """
    Visualise la comparaison des politiques
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique des récompenses
    policies = list(results.keys())
    rewards = [results[p]['avg_reward'] for p in policies]
    reward_stds = [results[p]['std_reward'] for p in policies]
    
    bars1 = ax1.bar(policies, rewards, yerr=reward_stds, capsize=5, alpha=0.7)
    ax1.set_ylabel("Récompense moyenne")
    ax1.set_title("Comparaison des récompenses")
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars1, rewards):
        if val == max(rewards):
            bar.set_color('green')
        elif val == min(rewards):
            bar.set_color('red')
    
    # Graphique des temps d'attente
    waitings = [results[p]['avg_waiting'] for p in policies]
    waiting_stds = [results[p]['std_waiting'] for p in policies]
    
    bars2 = ax2.bar(policies, waitings, yerr=waiting_stds, capsize=5, alpha=0.7)
    ax2.set_ylabel("Nombre moyen de véhicules en attente")
    ax2.set_title("Comparaison des files d'attente")
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars2, waitings):
        if val == min(waitings):
            bar.set_color('green')
        elif val == max(waitings):
            bar.set_color('red')
    
    plt.suptitle(f"Comparaison des politiques - {title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"comparison_{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


def run_scenarios():
    """
    Exécute la comparaison sur deux scénarios
    """
    scenarios = {
        'Trafic_equilibre': {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3},
        'Trafic_asymetrique': {'N': 0.5, 'S': 0.5, 'E': 0.1, 'O': 0.1}
    }
    
    all_results = {}
    
    for scenario_name, rates in scenarios.items():
        print(f"\n{'='*50}")
        print(f"Scénario: {scenario_name}")
        print(f"Taux d'arrivée: {rates}")
        print('='*50)
        
        results = compare_policies(rates, n_episodes=20)  # Réduit pour test rapide
        all_results[scenario_name] = results
        plot_comparison(results, title=scenario_name)
    
    return all_results


if __name__ == "__main__":
    print("=== Évaluation des politiques ===")
    results = run_scenarios()