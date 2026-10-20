"""
Everything is controlled by the True/False flags below. Turn on the one required
and run:   python main.py

    train                train a fresh agent from the fixed start (3,0), save
                         q_table.npy, and plot the learning curve
    run_experiments      compare 3 learning rates and 2 ε-decay strategies (plots)
    visualize            show the Q-value heatmaps and greedy policy for each phase
    evaluate             run 500 greedy episodes from (3,0) and print the summary
    random_init_compare  train a second agent from random start cells and compare
                         it against the fixed-start agent
    render_greedy        watch the trained agent play a few episodes (pygame window)

The submitted q_table.npy is the best agent (fixed start, gated exit): 500/500 exits.
  - To just watch it play:      leave render_greedy = True, then run main.py
  - Otherwise set render_greedy = False and evaluate = True
"""

# Imports:
# --------
import os
import numpy as np
from indiana_jones_env import IndianaJonesEnv
from q_learning_agent import train_q_learning, plot_training_curve, plot_comparison, visualize_q_table, evaluate_greedy


# ── What to run ────────────────────────────────────────────────────────────────
train            = False
run_experiments  = False    # compare learning rates and epsilon-decay strategies
visualize        = False
render           = True   # set True to watch the agent during training

# ── Base hyperparameters ───────────────────────────────────────────────────────
no_episodes   = 20_000
max_steps     = 200        # cap per episode — prevents infinite loops on non-terminal states
gamma         = 0.99       # high discount: reaching the far exit (7,7) from (3,0) matters
alpha         = 0.1        # learning rate
epsilon       = 1.0        # start fully random
epsilon_min   = 0.05       # never drop below 5% exploration
epsilon_decay = 0.9995     # slow decay — 8x8 grid needs extensive exploration

# ── Paths ──────────────────────────────────────────────────────────────────────
save_dir  = os.path.dirname(__file__)
save_path = os.path.join(save_dir, "q_table.npy")


# ── Train with default hyperparameters ────────────────────────────────────────
if train:
    
    print("=" * 60)
    print("Training Q-learning agent — Indiana Jones Tomb Escape")
    print("=" * 60)

    env = IndianaJonesEnv()
    rewards = train_q_learning(
        env               = env,
        no_episodes       = no_episodes,
        epsilon           = epsilon,
        epsilon_min       = epsilon_min,
        epsilon_decay     = epsilon_decay,
        alpha             = alpha,
        gamma             = gamma,
        max_steps         = max_steps,
        render            = render,
        q_table_save_path = save_path,
    )
    plot_training_curve(rewards, label=f"(α={alpha}, ε-decay={epsilon_decay})")


# ── Experiments ───────────────────────────────────────────────────────────────
if run_experiments:

    # Experiment 1: Compare three learning rates (α)
    # -----------------------------------------------
    print("=" * 60)
    print("Experiment 1: Comparing learning rates (α)")
    print("=" * 60)

    alpha_results = {}
    for a in [0.01, 0.1, 0.5]:
        print(f"\n── α = {a} ──")
        env = IndianaJonesEnv()
        rewards = train_q_learning(
            env               = env,
            no_episodes       = no_episodes,
            epsilon           = epsilon,
            epsilon_min       = epsilon_min,
            epsilon_decay     = epsilon_decay,
            alpha             = a,
            gamma             = gamma,
            max_steps         = max_steps,
            render            = False,
            q_table_save_path = os.path.join(save_dir, f"q_table_alpha_{a}.npy"),
        )
        alpha_results[f"α={a}"] = rewards

    plot_comparison(alpha_results, title="Experiment 1: Effect of Learning Rate (α)")

    # Experiment 2: Compare two epsilon-decay strategies
    # ---------------------------------------------------
    print("=" * 60)
    print("Experiment 2: Comparing ε-decay strategies")
    print("=" * 60)

    epsilon_results = {}

    # Strategy A: slow decay — explores for longer
    print(f"\n── Strategy A: Slow decay (ε-decay={epsilon_decay}) ──")
    env = IndianaJonesEnv()
    rewards_slow = train_q_learning(
        env               = env,
        no_episodes       = no_episodes,
        epsilon           = epsilon,
        epsilon_min       = epsilon_min,
        epsilon_decay     = epsilon_decay,
        alpha             = alpha,
        gamma             = gamma,
        max_steps         = max_steps,
        render            = False,
        q_table_save_path = os.path.join(save_dir, "q_table_slow_decay.npy"),
    )
    epsilon_results[f"Slow decay (ε-decay={epsilon_decay})"] = rewards_slow

    # Strategy B: fast decay — exploits earlier
    fast_decay = 0.995
    print(f"\n── Strategy B: Fast decay (ε-decay={fast_decay}) ──")
    env = IndianaJonesEnv()
    rewards_fast = train_q_learning(
        env               = env,
        no_episodes       = no_episodes,
        epsilon           = epsilon,
        epsilon_min       = epsilon_min,
        epsilon_decay     = fast_decay,
        alpha             = alpha,
        gamma             = gamma,
        max_steps         = max_steps,
        render            = False,
        q_table_save_path = os.path.join(save_dir, "q_table_fast_decay.npy"),
    )
    epsilon_results[f"Fast decay (ε-decay={fast_decay})"] = rewards_fast

    plot_comparison(epsilon_results, title="Experiment 2: Effect of ε-Decay Strategy")


# ── Visualize final Q-table and policy ────────────────────────────────────────
if visualize:
    print("=" * 60)
    print("Visualizing Q-table and learned policy")
    print("=" * 60)
    visualize_q_table(q_values_path=save_path)


# ── Greedy evaluation — proof that training worked ────────────────────────────
# Runs 100 episodes with ε=0 (pure exploitation, no random actions).
# Target: agent should reach the exit in the majority of episodes.
evaluate = False
if evaluate:
    env = IndianaJonesEnv()
    evaluate_greedy(
        env          = env,
        q_table_path = save_path,
        no_episodes  = 500,
        max_steps    = max_steps,
    )


# ── Random initialisation comparison ─────────────────────────────────────────
# Trains a second agent that starts from a random safe cell each episode,
# then compares its greedy evaluation against the fixed-start agent.
# This tests whether the policy generalises beyond the single start position.
random_init_compare = False
if random_init_compare:
    rand_save_path = os.path.join(save_dir, "q_table_random_init.npy")

    print("=" * 60)
    print("Training with random initialisation")
    print("=" * 60)
    env_rand = IndianaJonesEnv(random_init=True)
    rewards_rand = train_q_learning(
        env               = env_rand,
        no_episodes       = no_episodes,
        epsilon           = epsilon,
        epsilon_min       = epsilon_min,
        epsilon_decay     = epsilon_decay,
        alpha             = alpha,
        gamma             = gamma,
        max_steps         = max_steps,
        render            = False,
        q_table_save_path = rand_save_path,
    )

    # Plot the random init training curve
    plot_training_curve(rewards_rand, label=f"(Random init, α={alpha}, ε-decay={epsilon_decay})")

    # Visualize the random init Q-table and greedy policy for comparison
    print("=" * 60)
    print("Visualizing Random Init Q-table and learned policy")
    print("=" * 60)
    visualize_q_table(q_values_path=rand_save_path)

    # ── Side-by-side greedy evaluation ───────────────────────────────────────
    # Both agents are tested from random starts (same conditions).
    # Fixed-start agent: was trained only from (3,0) — tests generalisation.
    # Random-init agent: was trained from all safe cells — expected to do better.

    print("\n── Greedy evaluation: Fixed-start agent tested from RANDOM starts ──")
    env_fixed_rand = IndianaJonesEnv(random_init=True)
    evaluate_greedy(
        env          = env_fixed_rand,
        q_table_path = save_path,        # q_table.npy — trained from (3,0) only
        no_episodes  = 500,
        max_steps    = max_steps,
    )

    print("\n── Greedy evaluation: Random-init agent tested from RANDOM starts ──")
    env_rand_eval = IndianaJonesEnv(random_init=True)
    evaluate_greedy(
        env          = env_rand_eval,
        q_table_path = rand_save_path,   # q_table_random_init.npy — trained from all safe cells
        no_episodes  = 500,
        max_steps    = max_steps,
    )


# ── Visual render — watch the greedy agent play ───────────────────────────────
# Runs a small number of episodes with the pygame window open. Set no_episodes low (3-5) — it's slow.
render_greedy = True
if render_greedy:
    env_vis = IndianaJonesEnv()
    evaluate_greedy(
        env          = env_vis,
        q_table_path = save_path,
        no_episodes  = 3,
        max_steps    = max_steps,
        render       = True,
    )
