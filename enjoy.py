import os
import sys
import pygame
from environment import CarRacingEnv
from agent import DQNAgent

def enjoy():
    # Check if a model checkpoint was provided, default to best_model.pt
    model_path = "models/best_model.pt"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
    print(f"Loading model from: {model_path}")
    
    # Environment Setup (Human render mode)
    env = CarRacingEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Agent Setup
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Load model weights if they exist
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Model loaded successfully. Starting simulation.")
    else:
        print(f"WARNING: Model file '{model_path}' not found!")
        print("Starting in MANUAL driving mode. Press 'A' to switch to AI (which will be random since no model is loaded).")
        
    # Simulation control variables
    manual_mode = not os.path.exists(model_path) # Default to manual if no model exists
    running = True
    
    state, info = env.reset()
    
    print("\n--- Controls ---")
    print("  Arrow Keys / WASD : Drive manually (in manual mode)")
    print("  M                 : Switch to MANUAL Mode")
    print("  A                 : Switch to AI Mode")
    print("  R                 : Reset Track")
    print("  ESC               : Quit")
    
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    manual_mode = True
                    print("Switched to MANUAL MODE")
                elif event.key == pygame.K_a:
                    manual_mode = False
                    print("Switched to AI MODE")
                elif event.key == pygame.K_r:
                    state, info = env.reset()
                    print("Track Reset!")
                    
        if manual_mode:
            # Manual Driving Mode (read keys)
            keys = pygame.key.get_pressed()
            steer = 0
            accel = 0
            
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steer = -1
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steer = 1
                
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                accel = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                accel = -1
                
            # Map steering and acceleration inputs to the 9 discrete environment actions
            if accel == 1:
                action = 0 if steer == -1 else (2 if steer == 1 else 1)
            elif accel == -1:
                action = 6 if steer == -1 else (8 if steer == 1 else 7)
            else:
                action = 3 if steer == -1 else (5 if steer == 1 else 4)
        else:
            # AI Driving Mode
            action = agent.select_action(state, evaluate=True)
            
        # Step the environment
        state, reward, terminated, truncated, info = env.step(action)
        
        # Display feedback in console on collision
        if info["collided"]:
            print("Collision detected! Resetting track...")
            state, info = env.reset()
            
        if truncated:
            print("Car stagnated! Resetting track...")
            state, info = env.reset()
            
    env.close()

if __name__ == "__main__":
    enjoy()
