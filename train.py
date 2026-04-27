"""
Script d'entraînement pour l'agent Q-learning
avec visualisation de l'évolution de la table Q
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

from simulation import Intersection
from agent import QLearningAgent


def visualize_q_table(agent: QLearningAgent, 
                      episode: int, 
                      save_dir: str = "plots") -> None:
    """
    Visualise l'évolution de la table Q pour des états clés
    
    Args:
        agent: l'agent Q-learning
        episode: numéro de l'épisode (pour le nom du fichier)
        save_dir: dossier de sauvegarde des figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sélection d'états représentatifs pour visualisation
    # Chaque état = (N, S, E, O, phase)
    states_to_show = []
    labels = []
    
    # Cas 1: Files équilibrées
    for load in [0, 2, 5]:  # faible, moyen, chargé
        for phase in [0, 1]:
            state = (load, load, load, load, phase)
            states_to_show.append(state)
            labels.append(f"Eq{load}_ph{phase}")
    
    # Cas 2: Déséquilibre NS vs EO
    for imbalance in [(5,0), (4,1), (3,2)]:
        for phase in [0, 1]:
            state = (imbalance[0], imbalance[0], imbalance[1], imbalance[1], phase)
            states_to_show.append(state)
            labels.append(f"NS{imbalance[0]}_EO{imbalance[1]}_ph{phase}")
    
    # Cas 3: Une seule file chargée
    state = (5, 0, 0, 0, 0)
    states_to_show.append(state)
    labels.append("N5_only_ph0")
    
    state = (0, 0, 5, 0, 1)
    states_to_show.append(state)
    labels.append("E5_only_ph1")
    
    # Récupérer les Q-values
    q_maintain = []
    q_change = []
    
    for state in states_to_show:
        q0 = agent.Q[state][0]  # Maintenir
        q1 = agent.Q[state][1]  # Changer
        q_maintain.append(q0)
        q_change.append(q1)
    
    # Créer la visualisation
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(labels))
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
    
    # Ajouter des annotations pour les valeurs
    for bar in bars1:
        height = bar.get_height()
        if abs(height) > 0.1:
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    for bar in bars2:
        height = bar.get_height()
        if abs(height) > 0.1:
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
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
    Visualise la table Q sous forme de heatmap pour les phases
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Créer une heatmap pour la phase 0 (NS vert)
    q_matrix_phase0 = np.zeros((6, 6))  # 6x6 pour NS vs EO
    q_matrix_phase1 = np.zeros((6, 6))
    
    for ns in range(6):
        for eo in range(6):
            state0 = (ns, ns, eo, eo, 0)
            state1 = (ns, ns, eo, eo, 1)
            # Prendre la différence Q(change) - Q(maintain)
            q_matrix_phase0[ns, eo] = agent.Q[state0][1] - agent.Q[state0][0]
            q_matrix_phase1[ns, eo] = agent.Q[state1][1] - agent.Q[state1][0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap phase 0
    im1 = ax1.imshow(q_matrix_phase0, cmap='RdYlGn', aspect='auto', 
                     vmin=-10, vmax=10, interpolation='nearest')
    ax1.set_xlabel('Véhicules EO', fontsize=11)
    ax1.set_ylabel('Véhicules NS', fontsize=11)
    ax1.set_title(f'Phase NS vert (ép. {episode})\nPositif = changer', fontsize=11)
    ax1.set_xticks(range(6))
    ax1.set_yticks(range(6))
    
    # Heatmap phase 1
    im2 = ax2.imshow(q_matrix_phase1, cmap='RdYlGn', aspect='auto',
                     vmin=-10, vmax=10, interpolation='nearest')
    ax2.set_xlabel('Véhicules EO', fontsize=11)
    ax2.set_ylabel('Véhicules NS', fontsize=11)
    ax2.set_title(f'Phase EO vert (ép. {episode})\nPositif = changer', fontsize=11)
    ax2.set_xticks(range(6))
    ax2.set_yticks(range(6))
    
    plt.colorbar(im1, ax=ax1, label='Q(change) - Q(maintain)')
    plt.colorbar(im2, ax=ax2, label='Q(change) - Q(maintain)')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/q_heatmap_ep{episode:04d}.png", dpi=150)
    plt.close()


def train_agent(env: Intersection,
                agent: QLearningAgent,
                n_episodes: int = 500,
                steps_per_episode: int = 200,
                visualize_q: bool = True,
                save_every: int = 100,
                verbose: bool = True) -> Tuple[List[float], List[float]]:
    """
    Entraîne l'agent sur plusieurs épisodes
    
    Args:
        env: environnement du carrefour
        agent: agent Q-learning
        n_episodes: nombre d'épisodes d'entraînement
        steps_per_episode: durée de chaque épisode (pas de temps)
        visualize_q: activer la visualisation de la table Q
        save_every: fréquence de sauvegarde/viz (épisodes)
        verbose: afficher les logs
    
    Returns:
        (rewards_per_episode, epsilon_history)
    """
    rewards_per_episode = []
    epsilon_history = []
    best_reward = -float('inf')
    
    # Créer le dossier pour les visualisations
    if visualize_q:
        os.makedirs("plots", exist_ok=True)
    
    print("="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT Q-LEARNING")
    print(f"Paramètres: α={agent.alpha}, γ={agent.gamma}, "
          f"ε_start={1.0}, ε_min={agent.epsilon_min}, ε_decay={agent.epsilon_decay}")
    print(f"Épisodes: {n_episodes}, Pas par épisode: {steps_per_episode}")
    print("="*60)
    
    for episode in range(n_episodes):
        # Réinitialiser l'environnement
        env.reset()
        state_key = env.get_state_key()
        episode_reward = 0
        
        # Un épisode
        for step in range(steps_per_episode):
            # Choisir l'action selon ε-greedy
            action = agent.act(state_key)
            
            # Exécuter l'action
            reward, next_state_key = env.step(action)
            
            # Mettre à jour la table Q
            agent.learn(state_key, action, reward, next_state_key)
            
            # Mettre à jour l'état
            state_key = next_state_key
            episode_reward += reward
        
        # Fin d'épisode: décroître epsilon
        agent.decay_epsilon()
        rewards_per_episode.append(episode_reward)
        epsilon_history.append(agent.epsilon)
        
        # Sauvegarder le meilleur modèle
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("agent_best.pkl")
        
        # Visualisations périodiques
        if visualize_q and (episode + 1) % save_every == 0:
            visualize_q_table(agent, episode + 1)
            visualize_q_heatmap(agent, episode + 1)
        
        # Affichage des progrès
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:]) if episode >= 50 else np.mean(rewards_per_episode)
            print(f"Épisode {episode+1:4d}/{n_episodes} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Moyenne (50): {avg_reward:6.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Meilleur: {best_reward:6.1f}")
    
    # Visualisation finale
    if visualize_q:
        visualize_q_table(agent, n_episodes)
        visualize_q_heatmap(agent, n_episodes)
    
    # Sauvegarde finale
    agent.save("agent_final.pkl")
    
    print("="*60)
    print(f"ENTRAÎNEMENT TERMINÉ")
    print(f"Récompense moyenne (dernier épisode): {np.mean(rewards_per_episode[-50:]):.1f}")
    print(f"Meilleure récompense: {best_reward:.1f}")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    print("="*60)
    
    return rewards_per_episode, epsilon_history


def plot_training_curves(rewards: List[float], 
                         epsilon_history: List[float],
                         smooth: int = 20,
                         save_dir: str = "plots") -> None:
    """
    Trace les courbes d'apprentissage
    
    Args:
        rewards: liste des récompenses par épisode
        epsilon_history: historique d'epsilon
        smooth: fenêtre de lissage
        save_dir: dossier de sauvegarde
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Graphique 1: Récompenses
    episodes = range(1, len(rewards) + 1)
    
    # Courbe brute (légèrement transparente)
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Récompense brute')
    
    # Courbe lissée
    if len(rewards) > smooth:
        smoothed = np.convolve(rewards, np.ones(smooth)/smooth, mode='valid')
        ax1.plot(episodes[smooth-1:], smoothed, 'b-', linewidth=2, 
                label=f'Lissée (moyenne sur {smooth})')
    
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Récompense cumulée', fontsize=12)
    ax1.set_title('Courbe d\'apprentissage - Évolution des récompenses', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Epsilon
    ax2.plot(episodes, epsilon_history, 'r-', linewidth=2)
    ax2.set_xlabel('Épisode', fontsize=12)
    ax2.set_ylabel('Epsilon (ε)', fontsize=12)
    ax2.set_title('Décroissance de l\'exploration ε-greedy', fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.savefig(f"{save_dir}/training_curves.pdf", dpi=150)
    plt.show()
    
    print(f"\nCourbes sauvegardées dans {save_dir}/training_curves.png/pdf")


def train_with_params(arrival_rates: Dict[str, float],
                      alpha: float = 0.1,
                      gamma: float = 0.95,
                      epsilon_decay: float = 0.995,
                      n_episodes: int = 500,
                      label: str = "",
                      visualize: bool = True) -> Tuple[QLearningAgent, List[float]]:
    """
    Entraîne un agent avec des paramètres spécifiques
    
    Justification des paramètres:
    - alpha = 0.1 : taux d'apprentissage modéré, évite les oscillations
    - gamma = 0.95 : privilégie les récompenses à court terme (trafic change vite)
    - epsilon_decay = 0.995 : exploration forte au début, exploitation ensuite
    """
    # Créer l'environnement
    env = Intersection(arrival_rates=arrival_rates, max_queue=5)
    
    # Créer l'agent avec les paramètres justifiés
    agent = QLearningAgent(
        alpha=alpha,           # Justification: convergence stable
        gamma=gamma,          # Justification: horizon temporel court
        epsilon=1.0,          # Exploration maximale au début
        epsilon_min=0.01,     # Exploration minimale (5% d'actions aléatoires)
        epsilon_decay=epsilon_decay  # Décroissance progressive
    )
    
    # Entraînement
    rewards, epsilon_hist = train_agent(
        env, agent, 
        n_episodes=n_episodes, 
        visualize_q=visualize,
        verbose=True
    )
    
    # Sauvegarder l'agent
    filename = f"agent_{label}.pkl" if label else "agent.pkl"
    agent.save(filename)
    
    # Tracer les courbes
    plot_training_curves(rewards, epsilon_hist)
    
    return agent, rewards


def parameter_study():
    """
    Étude de l'influence des paramètres α et ε_decay
    """
    arrival_rates = {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3}
    
    # Paramètres à tester
    alphas = [0.05, 0.1, 0.3, 0.5]
    decays = [0.99, 0.995, 0.999]
    
    results = {}
    
    for alpha in alphas:
        for decay in decays:
            print(f"\n{'='*40}")
            print(f"Test: α={alpha}, ε_decay={decay}")
            print('='*40)
            
            env = Intersection(arrival_rates=arrival_rates, max_queue=5)
            agent = QLearningAgent(alpha=alpha, gamma=0.95, 
                                   epsilon_decay=decay, epsilon_min=0.01)
            
            rewards, _ = train_agent(env, agent, n_episodes=200, 
                                     visualize_q=False, verbose=False)
            
            avg_reward = np.mean(rewards[-50:])
            results[(alpha, decay)] = avg_reward
            print(f"Récompense moyenne (50 derniers): {avg_reward:.1f}")
    
    # Afficher les résultats
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ÉTUDE DES PARAMÈTRES")
    print("="*60)
    print("(α, ε_decay) → Récompense moyenne (50 derniers épisodes)")
    print("-"*50)
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    for (alpha, decay), reward in sorted_results:
        print(f"α={alpha:.2f}, ε_decay={decay:.3f} → {reward:.1f}")
    
    print(f"\nMeilleur paramètre: α={sorted_results[0][0][0]}, "
          f"ε_decay={sorted_results[0][0][1]} → {sorted_results[0][1]:.1f}")
    
    return results


if __name__ == "__main__":
    # Test rapide avec paramètres par défaut
    print("=== TEST D'ENTRAÎNEMENT RAPIDE ===")
    
    agent, rewards = train_with_params(
        {'N': 0.3, 'S': 0.3, 'E': 0.3, 'O': 0.3},
        n_episodes=100,
        visualize=True
    )
    
    print(f"\nEntraînement terminé!")
    print(f"Récompense moyenne (10 derniers épisodes): {np.mean(rewards[-10:]):.1f}")