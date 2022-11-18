import numpy as np

# simulation settings
SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 60

NUM_EVALUATIONS = 20000
NUM_RUNS = 10

# DE settings
DE_PARAMS = [
    (100, 0.9, 0.8),
    (100, 0.9, 0.2),
]  # population_size, crossover_probability, differential_weight

# Graph settings
GRAPH_PARAMS = (0.2,)  # standard deviation

# Terrain parameters
TERRAIN_SIZE = (5.0, 5.0)
TERRAIN_GRANULARITY = 0.5
RUGGEDNESS_RANGE = np.arange(0.0, 0.2, 0.1)  # np.arange(0.0, 0.5, 0.1)
BOWLNESS_RANGE = np.arange(0.0, 0.8, 0.4)  # np.arange(0.0, 2.0, 0.4)
