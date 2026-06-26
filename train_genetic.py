import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from genetic_environment import GeneticCarRacingEnv
from model import DQN  # We reuse the same model architecture

def get_chromosome(model):
    """
    Flattens all weights and biases of a PyTorch model into a 1D numpy array.
    """
    with torch.no_grad():
        params = [p.cpu().numpy().flatten() for p in model.parameters()]
        return np.concatenate(params)

def set_chromosome(model, chromosome):
    """
    Loads a 1D numpy array back into a PyTorch model's weights and biases.
    """
    with torch.no_grad():
        idx = 0
        for p in model.parameters():
            shape = p.shape
            size = p.numel()
            flat_p = chromosome[idx : idx + size]
            p.copy_(torch.tensor(flat_p.reshape(shape), dtype=torch.float32))
            idx += size

def mutate(chromosome, mutation_rate=0.15, mutation_scale=0.1):
    """
    Adds Gaussian noise to random weights of the chromosome.
    """
    mutated = chromosome.copy()
    mask = np.random.rand(len(chromosome)) < mutation_rate
    noise = np.random.normal(0, mutation_scale, size=np.sum(mask))
    mutated[mask] += noise
    return mutated

def crossover(parent1, parent2):
    """
    Performs uniform crossover between two parent chromosomes.
    """
    mask = np.random.rand(len(parent1)) < 0.5
    return np.where(mask, parent1, parent2)

def train_genetic():
    os.makedirs("models", exist_ok=True)
    
    # Check flags
    render_mode = None
    if "--render" in sys.argv:
        render_mode = "human"
        sys.argv.remove("--render")
        
    num_obstacles = 0
    random_start = True
    num_generations = 100
    for arg in sys.argv[:]:
        if arg.startswith("--obstacles="):
            try:
                num_obstacles = int(arg.split("=")[1])
            except ValueError:
                pass
            sys.argv.remove(arg)
        elif arg.startswith("--generations="):
            try:
                num_generations = int(arg.split("=")[1])
            except ValueError:
                pass
            sys.argv.remove(arg)
        elif arg == "--fixed-start":
            random_start = False
            sys.argv.remove(arg)
            
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
    # Hyperparameters
    pop_size = 40
    elite_size = 4        # Top 10% kept directly
    parents_pool_size = 16 # Top 40% used for reproduction
    
    mutation_rate = 0.15
    mutation_scale = 0.1
    
    # Env setup
    env = GeneticCarRacingEnv(pop_size=pop_size, render_mode=render_mode, num_obstacles=num_obstacles)

    state_dim = env.observation_space.shape[0]
    action_dim = 9
    
    # Initialize networks
    networks = [DQN(state_dim, action_dim) for _ in range(pop_size)]
    
    start_gen = 1
    pretrained_chromosome = None
    
    if model_path:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            temp_model = DQN(state_dim, action_dim)
            temp_model.load_state_dict(checkpoint['online_net_state_dict'])
            pretrained_chromosome = get_chromosome(temp_model)
            print(f"Loaded pretrained model from: {model_path} to seed the population.")
            
            # Try to extract starting generation if filename contains gen number
            base = os.path.basename(model_path)
            if "checkpoint_gen_" in base:
                try:
                    start_gen = int(base.replace("checkpoint_gen_", "").replace(".pt", "")) + 1
                    print(f"Resuming starting from Generation {start_gen}")
                except ValueError:
                    pass
        else:
            print(f"ERROR: Pretrained model file '{model_path}' not found! Starting with a random population.")
            
    # Generate initial chromosomes
    chromosomes = []
    if pretrained_chromosome is not None:
        # Clone 0 is the exact champion
        chromosomes.append(pretrained_chromosome)
        # Clones 1 to N-1 are mutated versions of the champion
        for i in range(1, pop_size):
            # Scale mutation for diversity (half close-range, half wide-range)
            scale = mutation_scale * (2.0 if i % 2 == 0 else 0.5)
            mutated_clone = mutate(pretrained_chromosome, mutation_rate=mutation_rate, mutation_scale=scale)
            chromosomes.append(mutated_clone)
    else:
        # Random initialization
        chromosomes = [get_chromosome(net) for net in networks]
        
    best_fitness_history = []
    avg_fitness_history = []
    
    print("\n--- Starting Genetic Algorithm Training ---")
    print(f"Population size: {pop_size} | Running for: {num_generations} generations (Starting at: {start_gen})")
    print(f"Render mode: {render_mode if render_mode else 'Headless (fast)'}")
    if not render_mode:
        print("Tip: Run 'python train_genetic.py --render' to watch the training visually!")
        
    for gen_idx in range(num_generations):
        gen = start_gen + gen_idx
        # Reset env
        obs = env.reset(generation=gen, random_start=random_start)

        done = False
        
        # Load chromosomes into models
        for i in range(pop_size):
            set_chromosome(networks[i], chromosomes[i])
            networks[i].eval()
            
        while not done:
            actions = []
            for i in range(pop_size):
                if not env.active[i]:
                    actions.append(4) # Do nothing if dead
                    continue
                    
                # Forward pass (CPU numpy/torch)
                state_t = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_vals = networks[i](state_t)
                    action = q_vals.argmax(dim=1).item()
                actions.append(action)
                
            obs, fitnesses, done, _ = env.step(actions)
            
        # Generation completed
        max_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        leader_cp = np.max(env.current_checkpoint)
        
        best_fitness_history.append(max_fit)
        avg_fitness_history.append(avg_fit)
        
        print(f"Gen {gen:03d} | Max Fitness: {max_fit:7.1f} | Avg Fitness: {avg_fit:7.1f} | Best Checkpoint: {leader_cp:3d}/{env.num_checkpoints}")
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        # Save absolute best network of this generation
        best_index = sorted_indices[0]
        # Save model weights to file
        torch.save({
            'online_net_state_dict': networks[best_index].state_dict(),
            'metadata': {
                'best_avg_reward': float(max_fit),
                'best_avg_checkpoints': int(leader_cp)
            }
        }, "models/genetic_best.pt")
        
        # Periodically save generation checkpoints
        if gen % 20 == 0:
            torch.save({
                'online_net_state_dict': networks[best_index].state_dict(),
                'metadata': {
                    'best_avg_reward': float(max_fit),
                    'best_avg_checkpoints': int(leader_cp)
                }
            }, f"models/checkpoint_gen_{gen}.pt")
            print(f"Saved checkpoint models/checkpoint_gen_{gen}.pt")
        
        # Selection for next generation
        new_chromosomes = []
        
        # 1. Elitism: Keep top K individuals unchanged
        for idx in sorted_indices[:elite_size]:
            new_chromosomes.append(chromosomes[idx])
            
        # 2. Reproduction: Select parents from top P pool
        parent_indices = sorted_indices[:parents_pool_size]
        
        while len(new_chromosomes) < pop_size:
            # Select parents randomly from pool
            p1_idx, p2_idx = np.random.choice(parent_indices, size=2, replace=False)
            p1 = chromosomes[p1_idx]
            p2 = chromosomes[p2_idx]
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            # Gradually reduce mutation scale over the course of this training run to fine-tune
            progress_ratio = gen_idx / num_generations
            scale = mutation_scale * (1.0 - progress_ratio * 0.5)
            child = mutate(child, mutation_rate, scale)
            
            new_chromosomes.append(child)
            
        chromosomes = new_chromosomes
        
    env.close()
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history, label="Best Fitness", color="blue", linewidth=2)
    plt.plot(avg_fitness_history, label="Avg Fitness", color="orange", linestyle="--")
    plt.title("Genetic Algorithm Training Progress")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig("models/genetic_progress.png")
    plt.close()
    print("Saved training curve to models/genetic_progress.png")

if __name__ == "__main__":
    train_genetic()
