import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import CarRacingEnv
from agent import DQNAgent

def train():
    # Directories
    os.makedirs("models", exist_ok=True)
    
    # Check if a model checkpoint was provided to resume training
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Hyperparameters
    num_episodes = 400
    batch_size = 128
    learning_rate = 5e-4
    gamma = 0.98
    epsilon_decay = 25000 # decay speed (in agent steps)
    
    # Environment & Agent
    env = CarRacingEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Metrics and tracking values
    episode_rewards = []
    episode_checkpoints = []
    episode_steps = []
    best_avg_reward = -float('inf')
    best_avg_checkpoints = 0
    
    if model_path:
        if os.path.exists(model_path):
            metadata = agent.load(model_path)
            if metadata:
                best_avg_reward = metadata.get('best_avg_reward', -float('inf'))
                best_avg_checkpoints = metadata.get('best_avg_checkpoints', 0)
                print(f"Loaded metadata - Best Avg Reward: {best_avg_reward:.2f}, Best Avg CPs: {best_avg_checkpoints:.2f}")
            print(f"Resuming training from checkpoint: {model_path}")
        else:
            print(f"ERROR: Checkpoint file '{model_path}' not found! Starting training from scratch.")
    
    print("\n--- Starting Training ---")
    print(f"State space: {state_dim}, Action space: {action_dim}")
    print(f"Total episodes: {num_episodes}")
    
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        max_cp = 0
        
        while True:
            # Select action
            action = agent.select_action(state, evaluate=False)
            
            # Step the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store in replay buffer
            # PyTorch models prefer terminated for DQN targets, let's pass terminated or truncated
            agent.memory.push(state, action, reward, next_state, terminated)
            
            # Learn
            loss = agent.update()
            
            state = next_state
            total_reward += reward
            steps += 1
            max_cp = max(max_cp, info["checkpoint"])
            
            if done:
                break
                
        episode_rewards.append(total_reward)
        episode_checkpoints.append(max_cp)
        episode_steps.append(steps)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_checkpoints = np.mean(episode_checkpoints[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            
            print(f"Episode {episode:03d} | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Avg CPs: {avg_checkpoints:5.1f}/{env.num_checkpoints} | "
                  f"Avg Steps: {avg_steps:5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
            # Save the best model based on checkpoint completion
            if avg_checkpoints >= best_avg_checkpoints and avg_reward >= best_avg_reward - 10:
                best_avg_checkpoints = avg_checkpoints
                best_avg_reward = avg_reward
                agent.save("models/best_model.pt", metadata={
                    'best_avg_reward': float(best_avg_reward),
                    'best_avg_checkpoints': int(best_avg_checkpoints)
                })
                
        # Periodically save regular checkpoint
        if episode % 50 == 0:
            agent.save(f"models/checkpoint_{episode}.pt", metadata={
                'best_avg_reward': float(best_avg_reward),
                'best_avg_checkpoints': int(best_avg_checkpoints)
            })
            
    print("\n--- Training Finished! ---")
    agent.save("models/final_model.pt", metadata={
        'best_avg_reward': float(best_avg_reward),
        'best_avg_checkpoints': int(best_avg_checkpoints)
    })

    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Episode Reward", color="purple")
    # Running average
    running_avg_rewards = [np.mean(episode_rewards[max(0, i-20):i+1]) for i in range(len(episode_rewards))]
    plt.plot(running_avg_rewards, label="20-ep Running Avg", color="blue", linewidth=2)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_checkpoints, label="Max Checkpoint Reached", color="orange")
    running_avg_cps = [np.mean(episode_checkpoints[max(0, i-20):i+1]) for i in range(len(episode_checkpoints))]
    plt.plot(running_avg_cps, label="20-ep Running Avg", color="red", linewidth=2)
    plt.title("Max Checkpoint Reached")
    plt.xlabel("Episode")
    plt.ylabel("Checkpoint Index")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("models/training_progress.png")
    plt.close()
    print("Saved training curve to models/training_progress.png")

if __name__ == "__main__":
    train()
