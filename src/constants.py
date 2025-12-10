"""
Centralized configuration for the batch-aware RL scheduler.
"""

# ================================================================
# SECTION 1: PERFORMANCE PROFILE
# ================================================================
PERFORMANCE_PROFILE = {
    1: 0.0016378,
    2: 0.0015201,
    4: 0.0013424,
    8: 0.0016604,
    16: 0.0027963,
    32: 0.0053908,
    64: 0.0119798
}

# ================================================================
# SECTION 2: SENSOR TASK DEFINITIONS
# ================================================================
TASK_TYPES = [
    {'name': 'camera', 'deadline': 0.03, 'arrival_interval': 0.033},  
    {'name': 'radar', 'deadline': 0.02, 'arrival_interval': 0.020},  
    {'name': 'lidar', 'deadline': 0.05, 'arrival_interval': 0.100},
    {'name': 'audio', 'deadline': None, 'arrival_interval': None, 'random_deadline_range': (0.1, 0.5), 'random_arrival_range': (0.05, 0.3)}  # Random deadline and arrival for audio
]
NUM_TASK_TYPES = len(TASK_TYPES)

# Optional single-type mode
USE_SINGLE_TASK_TYPE = False

# Simulation step granularity
SIM_STEP_SECONDS = 0.01

# ================================================================
# SECTION 3: RL AGENT & TRAINING PARAMETERS
# ================================================================
TOTAL_TIMESTEPS = 100_000

# Learning rate for the RL agent's neural network.
LEARNING_RATE = 0.0001

# Discount factor for future rewards. 
GAMMA = 0.99

# The size of the replay buffer, where the agent stores past experiences.
BUFFER_SIZE = 10_000

# How many steps of experience to collect before starting to train the model.
LEARNING_STARTS = 1000


# ================================================================
# SECTION 4: ENVIRONMENT DEFINITION
# ================================================================
NUM_STATE_FEATURES = 12

# Action Space Definitions
ACTION_WAIT = 0
BATCH_SIZE_OPTIONS = [1, 2, 4, 8, 16, 32, 64]
NUM_ACTIONS = len(BATCH_SIZE_OPTIONS) + 1  # +1 for WAIT action


# ================================================================
# SECTION 5: ADAS-SPECIFIC PARAMETERS
# ================================================================
IMAGENETTE_PATH = "data/imagenette2"
SPEECH_COMMANDS_PATH = "data/speech_commands"  # Google Speech Commands Dataset path
INFERENCE_DEVICE = "cuda"  # "cuda" for GPU, "cpu" for CPU-only

# ================================================================
# SECTION 6: REWARD FUNCTION PARAMETERS
# ================================================================
# Reward for successfully completing a task before deadline
REWARD_TASK_SUCCESS = 2.0

# Penalty for missing a task's deadline
PENALTY_TASK_MISS = 10.0

# Penalty for a task expiring in the queue
PENALTY_QUEUE_EXPIRY = 15.0

# Latency penalty coefficient
LATENCY_PENALTY_COEFF = 0.3  # Increased for high-load scenario

# Batch efficiency bonus coefficient
BATCH_BONUS_COEFF = 0.5

# Invalid action penalties
PENALTY_EMPTY_QUEUE = 0.5
PENALTY_NODE_BUSY = 1.0
PENALTY_WAIT = 0.01  # Small penalty for waiting


# ================================================================
# SECTION 7: ENHANCED STATE SPACE PARAMETERS
# ================================================================
# Window size for calculating recent statistics
HISTORY_WINDOW_SIZE = 10

# ================================================================
# SECTION 8: MULTI-NODE CONFIGURATION
# ================================================================
NUM_EDGE_NODES = 3  # Number of simulated edge nodes

# Graph state configuration
USE_GRAPH_STATE = True  # Enable GNN-based graph state representation
MAX_TASKS_IN_GRAPH = 100  # Maximum number of tasks to include in graph

# GNN encoder configuration (only used when USE_GRAPH_STATE = True)
GNN_OUTPUT_DIM = 64  # Output dimension of GNN encoder
GNN_HIDDEN_DIM = 64  # Hidden dimension for GNN layers
GNN_NUM_LAYERS = 2   # Number of GNN convolution layers