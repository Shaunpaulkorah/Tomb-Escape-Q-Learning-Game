"""
Q-learning agent for the Indiana Jones tomb environment.

Provides training, greedy evaluation, and plotting/visualisation helpers.
This module is imported by main.py; it is not meant to be run directly.

The agent keeps 3 Q-tables (one per phase) so it can learn a separate policy for
each sub-goal: collect the 1st treasure -> collect the 2nd -> reach the exit.
The active phase is simply how many treasures have been collected so far (0, 1 or 2).
"""

# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     max_steps,
                     render,
                     q_table_save_path="q_table.npy"):
    """Train the agent with tabular Q-learning and save the 3 Q-tables to disk.

    Returns the list of total rewards per episode (used to plot the learning curve)."""

    # Three Q-tables stacked as (3, grid, grid, n_actions) — one per phase:
    #   phase 0: 0 treasures collected -> seek the first treasure
    #   phase 1: 1 treasure  collected -> seek the second treasure
    #   phase 2: both        collected -> seek the exit
    q_table = np.zeros((3, env.grid_size, env.grid_size, env.action_space.n))

    rewards_per_episode = []

    for episode in range(no_episodes):
        state, _ = env.reset()
        r, c = int(state[0]), int(state[1])
        total_reward = 0

        # max_steps caps the episode. Fire and scorpion pits are non-terminal, so
        # without a cap the agent could wander forever collecting penalties.
        for _ in range(max_steps):

            # Phase = treasures collected so far; this picks which Q-table to use.
            phase = len(env.collected_treasures)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()           # explore
            else:
                action = np.argmax(q_table[phase, r, c])     # exploit

            next_state, reward, terminated, truncated, _ = env.step(action)

            if render:
                env.render()

            nr, nc = int(next_state[0]), int(next_state[1])

            # Phase after the step — advances if a treasure was just collected.
            next_phase = len(env.collected_treasures)

            total_reward += reward # Accumulate reward for this episode for logging.

            # Q(s,a) = (1-alpha)*Q(s,a) + alpha*[ r + gamma * max Q_next(s') ]
            # The target reads next_phase's table, so when a treasure is collected the
            # future value is judged under the new sub-goal. That is what links the
            # three tables together and lets the +8/+8/+20 rewards propagate back.
            if terminated:
                target = reward # No future value if the episode ended (exit or death). S' is terminal. 
                                # target equation part of γ·max_a' Q(s', a') becomes zero, with only reward remaining.
            else:
                target = reward + gamma * np.max(q_table[next_phase, nr, nc])

            q_table[phase, r, c, action] = (1 - alpha) * q_table[phase, r, c, action] + \
                alpha * target

            r, c = nr, nc

            if terminated or truncated:
                break

        # Reduce exploration as the agent learns.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_per_episode.append(total_reward)

        print(f"  Episode {episode + 1:>6}/{no_episodes} | ε={epsilon:.4f} | Total reward: {total_reward:.2f}")

    env.close()
    print("Training finished.\n")

    # Save all 3 Q-tables in one file, shape (3, 8, 8, 4).
    np.save(q_table_save_path, q_table)
    print(f"Q-tables saved → {q_table_save_path}  (shape: {q_table.shape})\n")

    return rewards_per_episode


def evaluate_greedy(env, q_table_path, no_episodes=100, max_steps=200, render=False):
    """Run the trained agent greedily (ε=0) and report exits / deaths / timeouts.

    Uses the correct phase's Q-table at every step, so the agent follows the full
    plan: collect 1st treasure -> collect 2nd -> reach the exit."""

    q_table = np.load(q_table_path)   # shape (3, 8, 8, 4)

    exits    = 0   # reached the exit
    deaths   = 0   # hit a boulder or mummy
    timeouts = 0   # ran out of steps

    print(f"\n{'='*60}")
    print(f"Greedy Evaluation — {no_episodes} episodes, ε=0")
    print(f"{'='*60}")

    for episode in range(no_episodes):
        state, _ = env.reset()
        r, c = int(state[0]), int(state[1])
        total_reward = 0

        for step in range(max_steps):
            phase = len(env.collected_treasures)

            # Always take the best known action — no randomness.
            action = np.argmax(q_table[phase, r, c])
            next_state, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            nr, nc = int(next_state[0]), int(next_state[1])
            total_reward += reward
            r, c = nr, nc

            if terminated:
                if reward > 0:
                    exits += 1
                    outcome = "EXIT"
                else:
                    deaths += 1
                    outcome = "DEATH"
                break
        else:
            timeouts += 1
            outcome = "TIMEOUT"

        print(f"  Ep {episode+1:>3} | {outcome:<7} | Steps: {step+1:>3} | Reward: {total_reward:>7.2f}")

    env.close()

    print(f"\n── Summary ({'─'*38})")
    print(f"  Exits    : {exits:>3} / {no_episodes}  ({100*exits/no_episodes:.0f}%)")
    print(f"  Deaths   : {deaths:>3} / {no_episodes}  ({100*deaths/no_episodes:.0f}%)")
    print(f"  Timeouts : {timeouts:>3} / {no_episodes}  ({100*timeouts/no_episodes:.0f}%)")
    print(f"{'='*60}\n")

    return exits, deaths, timeouts


def plot_training_curve(rewards_per_episode, label="", window=200):
    """Plot the per-episode reward with a moving-average overlay."""
    plt.figure(figsize=(11, 4))
    plt.plot(rewards_per_episode, alpha=0.2, color='steelblue')
    if len(rewards_per_episode) >= window:
        moving_avg = np.convolve(rewards_per_episode, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rewards_per_episode)), moving_avg,
                 color='darkorange', linewidth=2, label=f'{window}-ep moving avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Reward Curve — Indiana Jones Q-Learning  {label}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison(results_dict, title="Experiment Comparison", window=200):
    """Overlay several training runs (e.g. different α or ε-decay) on one plot."""
    plt.figure(figsize=(12, 5))
    for label, rewards in results_dict.items():
        if len(rewards) >= window:
            avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(rewards)), avg, linewidth=2, label=label)
    plt.xlabel('Episode')
    plt.ylabel(f'Avg Reward ({window}-ep window)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_q_table(q_values_path="q_table.npy"):
    """Show, for each phase, the per-action Q-value heatmaps and the greedy policy
    (arrows + max Q-value). Terminal cells are masked and labelled."""

    exit_pos      = (7, 7)
    boulder_traps = [(0, 4), (2, 2), (5, 6)]
    mummy_guards  = [(4, 5), (7, 4)]
    terminal_states = [exit_pos] + boulder_traps + mummy_guards

    actions       = ["Up", "Down", "Left", "Right"]
    action_arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    phase_labels  = [
        "Phase 0 — Seek First Treasure",
        "Phase 1 — Seek Second Treasure",
        "Phase 2 — Seek Exit",
    ]

    try:
        q_table = np.load(q_values_path)   # shape (3, 8, 8, 4)

        terminal_mask = np.zeros(q_table.shape[1:3], dtype=bool)
        for s in terminal_states:
            terminal_mask[s] = True

        for phase in range(3):

            # Plot 1: one Q-value heatmap per action.
            _, axes = plt.subplots(1, 4, figsize=(24, 6))
            for i, action in enumerate(actions):
                ax = axes[i]
                sns.heatmap(q_table[phase, :, :, i].copy(), annot=True, fmt=".2f",
                            cmap="viridis", ax=ax, cbar=False, mask=terminal_mask,
                            annot_kws={"size": 7}, linewidths=0.3, linecolor="white")

                ax.text(exit_pos[1] + 0.5, exit_pos[0] + 0.5, 'EXIT', color='lime',
                        ha='center', va='center', weight='bold', fontsize=9)
                for pos in boulder_traps:
                    ax.text(pos[1] + 0.5, pos[0] + 0.5, 'BOUL', color='red',
                            ha='center', va='center', weight='bold', fontsize=9)
                for pos in mummy_guards:
                    ax.text(pos[1] + 0.5, pos[0] + 0.5, 'MUMM', color='red',
                            ha='center', va='center', weight='bold', fontsize=9)

                ax.set_title(f'Action: {action}')
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')

            plt.suptitle(f'Q-values per Action — {phase_labels[phase]}', fontsize=13)
            plt.tight_layout()
            plt.show()

            # Plot 2: greedy policy (arrows) over the max Q-value heatmap.
            best_actions  = np.argmax(q_table[phase], axis=2)
            best_q_values = np.max(q_table[phase], axis=2)

            _, ax = plt.subplots(figsize=(9, 8))
            sns.heatmap(best_q_values, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=True, mask=terminal_mask,
                        annot_kws={"size": 7}, linewidths=0.3, linecolor="white")

            for row in range(q_table.shape[1]):
                for col in range(q_table.shape[2]):
                    pos = (row, col)
                    if pos == exit_pos:
                        ax.text(col + 0.5, row + 0.5, 'EXIT', color='lime',
                                ha='center', va='center', weight='bold', fontsize=11)
                    elif pos in boulder_traps:
                        ax.text(col + 0.5, row + 0.5, 'BOUL', color='red',
                                ha='center', va='center', weight='bold', fontsize=10)
                    elif pos in mummy_guards:
                        ax.text(col + 0.5, row + 0.5, 'MUMM', color='red',
                                ha='center', va='center', weight='bold', fontsize=10)
                    else:
                        ax.text(col + 0.5, row + 0.25, action_arrows[best_actions[row, col]],
                                color='white', ha='center', va='center', weight='bold', fontsize=17)

            ax.set_title(f'Learned Greedy Policy — {phase_labels[phase]}', fontsize=13)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            plt.tight_layout()
            plt.show()

    except FileNotFoundError:
        print("No saved Q-table found. Train the agent first or check the path.")
