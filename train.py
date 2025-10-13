import os
from stable_baselines3 import DQN

# Import our custom environment and constants
from src.environment import SchedulingEnv
import src.constants as c

# --- 1. SETUP: Create directories to save results ---
# Create a directory to save the trained model and logs
results_path = "results"
os.makedirs(results_path, exist_ok=True)
print("--> Setup: 'results' directory created.")

# --- 2. CREATE THE ENVIRONMENT ---
# Instantiate our custom-made simulation world
env = SchedulingEnv()
print("--> Environment: Custom SchedulingEnv created.")

# --- 3. CREATE THE RL AGENT ---
# We choose the DQN algorithm, which is well-suited for discrete action spaces.
# 'MlpPolicy' means we use a standard Multi-Layer Perceptron (a type of ANN)
# as the agent's brain.
# We pass all the hyperparameters from our constants file.
# verbose=1 will print out the training progress.
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=c.LEARNING_RATE,
    buffer_size=c.BUFFER_SIZE,
    learning_starts=c.LEARNING_STARTS,
    gamma=c.GAMMA,
    verbose=1
)
print("--> Agent: DQN model created with parameters from constants.py.")

# --- 4. START TRAINING ---
# This is where the magic happens. The agent will now interact with the
# environment for the specified number of steps, learning from its actions.
print("\n--- Starting Training ---")
model.learn(total_timesteps=c.TOTAL_TIMESTEPS)
print("--- Training Finished ---\n")

# --- 5. SAVE THE TRAINED MODEL ---
# Save the agent's learned brain ("policy") to a file.
# We can later load this file in evaluate.py to test its performance.
model_save_path = os.path.join(results_path, f"dqn_scheduler_{c.TOTAL_TIMESTEPS}_steps.zip")
model.save(model_save_path)

print(f"--> Model saved to: {model_save_path}")

# --- 6. CLEAN UP ---
env.close()