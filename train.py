"""
Script d'entraînement pour l'agent Q-Learning
Inclut la visualisation de la table Q et les courbes de convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

from simulation import Intersection
from agent import QLearningAgent


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION DE LA TABLE Q
# ══════════════════════════════════════════════════════════════════════════════

def visualize_q_table(agent: QLearningAgent,
                       episode: int,
                       save_dir: str = "plots") -> None:
    """
    Visualise les valeurs Q pour un ensemble d'états représentatifs.
    Les clés d'état ont 6 éléments : (qN, qS, qE, qO, phase, in_orange=0)
    """
    os.makedirs(save_dir, exist_ok=True)

    states_to_show = []
    labels         = []

    # Cas 1 : Files équilibrées (in_orange = 0)
    for load in [0, 2, 5]:
        for phase in [0, 1]:
            state = (load, load, load, load, phase, 0)
            states_to_show.append(state)
            labels.append(f"Eq{load}_ph{phase}")

    # Cas 2 : Déséquilibre NS vs EO
    for (ns, eo) in [(5, 0), (4, 1), (3, 2)]:
        for phase in [0, 1]:
            state = (ns, ns, eo, eo, phase, 0)
            states_to_show.append(state)
            labels.append(f"NS{ns}_EO{eo}_ph{phase}")

    # Cas 3 : Une seule direction chargée
    states_to_show.append((5, 0, 0, 0, 0, 0))
    labels.append("N5_only_ph0")
    states_to_show.append((0, 0, 5, 0, 1, 0))
    labels.append("E5_only_ph1")

    # Récupérer les valeurs Q
    q_maintain = [agent.Q[s][0] for s in states_to_show]
    q_change   = [agent.Q[s][1] for s in states_to_show]

    # Tracer
    fig, ax = plt.subplots(figsize=(14, 7))
    x     = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, q_maintain, width,
                   label='Action: MAINtenir la phase',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, q_change, width,
                   label='Action: CHANGER de phase',
                   color='coral', alpha=0.8)

    ax.set_xlabel('États représentatifs', fontsize=12)
    ax.set_ylabel('Valeur Q', fontsize=12)
    ax.set_title(f'Évolution de la table Q - Épisode {episode}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.5:
                ax.annotate(f'{h:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/q_table_ep{episode:04d}.png", dpi=150)
    plt.close()
    print(f"  → Table Q visualisée (épisode {episode})")


def visualize_q_heatmap(agent: QLearningAgent,
                         episode: int,
                         save_dir: str = "plots") -> None:
    """
    Heatmap de Q(changer) - Q(maintenir) pour chaque phase.
    Clés d'état à 6 éléments (in_orange = 0).
    Vert = préférer changer / Rouge = préférer maintenir.
    """
    os.makedirs(save_dir, exist_ok=True)

    q_matrix_phase0 = np.zeros((6, 6))
    q_matrix_phase1 = np.zeros((6, 6))

    for ns in range(6):
        for eo in range(6):
            s0 = (ns, ns, eo, eo, 0, 0)   # phase NS, pas d'orange
            s1 = (ns, ns, eo, eo, 1, 0)   # phase EO, pas d'orange
            q_matrix_phase0[ns, eo] = agent.Q[s0][1] - agent.Q[s0][0]
            q_matrix_phase1[ns, eo] = agent.Q[s1][1] - agent.Q[s1][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, matrix, title in [
        (ax1, q_matrix_phase0, f'Phase NS vert (ép. {episode})\nPositif = changer'),
        (ax2, q_matrix_phase1, f'Phase EO vert (ép. {episode})\nPositif = changer'),
    ]:
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                       vmin=-10, vmax=10, interpolation='nearest')
        ax.set_xlabel('Véhicules EO', fontsize=11)
        ax.set_ylabel('Véhicules NS', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        plt.colorbar(im, ax=ax, label='Q(change) − Q(maintain)')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/q_heatmap_ep{episode:04d}.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def train_agent(env: Intersection,
                agent: QLearningAgent,
                n_episodes: int       = 500,
                steps_per_episode: int = 200,
                visualize_q: bool     = True,
                save_every: int       = 100,
                verbose: bool         = True) -> Tuple[List[float], List[float]]:
    """
    Boucle principale d'entraînement Q-Learning.

    À chaque épisode :
      1. Réinitialiser l'environnement
      2. Pour chaque pas : observer → agir → recevoir récompense → mettre à jour Q
      3. Décroître ε

    Returns:
        (rewards_per_episode, epsilon_history)
    """
    rewards_per_episode = []
    best_reward         = -float('inf')

    os.makedirs("plots", exist_ok=True)

    print("=" * 60)
    print("DÉBUT DE L'ENTRAÎNEMENT Q-LEARNING")
    print(f"Paramètres: α={agent.alpha}, γ={agent.gamma}, "
          f"ε_start=1.0, ε_min={agent.epsilon_min}, ε_decay={agent.epsilon_decay}")
    print(f"Épisodes: {n_episodes}, Pas par épisode: {steps_per_episode}")
    print("=" * 60)

    for episode in range(n_episodes):
        env.reset()
        state_key      = env.get_state_key()
        episode_reward = 0

        for _ in range(steps_per_episode):
            action                      = agent.act(state_key)
            reward, next_state_key      = env.step(action)
            agent.learn(state_key, action, reward, next_state_key)
            state_key                   = next_state_key
            episode_reward             += reward

        # Fin d'épisode
        agent.decay_epsilon()
        rewards_per_episode.append(episode_reward)

        # Sauvegarder le meilleur modèle
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("agent_best.pkl")

        # Visualisations périodiques
        if visualize_q and (episode + 1) % save_every == 0:
            visualize_q_table(agent, episode + 1)
            visualize_q_heatmap(agent, episode + 1)

        # Logs
        if verbose and (episode + 1) % 50 == 0:
            window    = rewards_per_episode[-50:]
            avg       = np.mean(window)
            print(f"Épisode {episode+1:4d}/{n_episodes} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Moyenne (50): {avg:7.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Meilleur: {best_reward:7.1f}")

    # Visualisation et sauvegarde finales
    if visualize_q:
        visualize_q_table(agent, n_episodes)
        visualize_q_heatmap(agent, n_episodes)

    agent.save("agent_final.pkl")

    print("=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"Récompense moyenne (50 derniers): {np.mean(rewards_per_episode[-50:]):.1f}")
    print(f"Meilleure récompense: {best_reward:.1f}")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    print("=" * 60)

    return rewards_per_episode, agent.epsilon_history


# ══════════════════════════════════════════════════════════════════════════════
# COURBES D'APPRENTISSAGE
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(rewards: List[float],
                          epsilon_history: List[float],
                          smooth: int    = 20,
                          save_dir: str  = "plots") -> None:
    """Trace et sauvegarde les courbes de convergence."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    episodes = range(1, len(rewards) + 1)

    # Récompenses
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Récompense brute')
    if len(rewards) > smooth:
        smoothed = np.convolve(rewards, np.ones(smooth) / smooth, mode='valid')
        ax1.plot(episodes[smooth - 1:], smoothed, 'b-', linewidth=2,
                 label=f'Lissée (moyenne sur {smooth})')
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Récompense cumulée', fontsize=12)
    ax1.set_title("Courbe d'apprentissage - Évolution des récompenses", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Epsilon
    ax2.plot(episodes, epsilon_history, 'r-', linewidth=2)
    ax2.set_xlabel('Épisode', fontsize=12)
    ax2.set_ylabel('Epsilon (ε)', fontsize=12)
    ax2.set_title("Décroissance de l'exploration ε-greedy", fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.savefig(f"{save_dir}/training_curves.pdf", dpi=150)
    plt.show()
    print(f"\nCourbes sauvegardées dans {save_dir}/training_curves.png/pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def train_with_params(arrival_rates: Dict[str, float],
                       alpha: float         = 0.1,
                       gamma: float         = 0.95,
                       epsilon_decay: float = 0.995,
                       n_episodes: int      = 500,
                       label: str           = "",
                       visualize: bool      = True) -> Tuple[QLearningAgent, List[float]]:
    """
    Entraîne un agent avec des paramètres donnés et sauvegarde le résultat.
    """
    env   = Intersection(arrival_rates=arrival_rates, max_queue=5)
    agent = QLearningAgent(alpha=alpha, gamma=gamma,
                           epsilon=1.0, epsilon_min=0.01,
                           epsilon_decay=epsilon_decay)

    rewards, epsilon_hist = train_agent(env, agent,
                                        n_episodes=n_episodes,
                                        visualize_q=visualize,
                                        verbose=True)

    filename = f"agent_{label}.pkl" if label else "agent_final.pkl"
    agent.save(filename)

    plot_training_curves(rewards, epsilon_hist)
    return agent, rewards


if __name__ == "__main__":
    print("=== ENTRAÎNEMENT RAPIDE (TEST) ===")
    agent, rewards = train_with_params(
        {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3},
        n_episodes=100, visualize=True
    )
    print(f"Récompense moyenne (10 derniers): {np.mean(rewards[-10:]):.1f}")
