import numpy as np

# simulation settings
SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 60

NUM_EVALUATIONS = 500000

# DE settings
DE_PARAMS = [
    (100, 0.9, 0.8),
    (100, 0.9, 0.2),
    (10, 0.9, 0.2),
]  # population_size, crossover_probability, differential_weight

# Graph settings
GRAPH_PARAMS = [(0.05, 0.5), (0.05, 0.0)]  # standard deviation, migration_probability

# Terrain parameters
TERRAIN_SIZE = (5.0, 5.0)
TERRAIN_GRANULARITY = 0.5
RUGGEDNESS_RANGE = np.arange(0.0, 0.5, 0.1)
BOWLNESS_RANGE = np.arange(0.0, 2.5, 0.5)

ROBOT_INITIAL_Z_OFFSET = 0.5
