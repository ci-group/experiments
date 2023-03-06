import numpy as np
import itertools

# simulation settings
SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 60

NUM_EVALUATIONS = 300000

# DE settings
DE_PARAMS = [
    (50, 0.9, 0.2, 1),
    (50, 0.9, 0.2, 125),
    (50, 0.9, 0.2, 5),
    (50, 0.9, 0.2, 25),
]  # population_size, crossover_probability, differential_weight, partition_size

CMAES_PARAMS = [
    (0.5, 1),
    (0.5, 125),
    (0.5, 5),
    (0.5, 25),
]  # initial std, partition_size

# Graph settings
GRAPH_STD = 0.05
GRAPH_PMIG = 0.5
GRAPH_ALPHA1 = 30.0
GRAPH_ALPHA2 = 3.0
THETA1S = [3.0]  # [0.0, 1.0, 2.0]
THETA2S = [0.125]  # [0.25, 0.5, float("inf")]
GRAPH_PARAMS = [
    (
        GRAPH_STD,
        0.0,
        GRAPH_ALPHA1,
        GRAPH_ALPHA2,
        0.0,
        1.0,
    )
    # (
    #     GRAPH_STD,
    #     GRAPH_PMIG,
    #     GRAPH_ALPHA1,
    #     GRAPH_ALPHA2,
    #     i[0],
    #     i[1],
    # )
    # for i in itertools.product(THETA1S, THETA2S)
]  # standard deviation, migration_probability, alpha1, alpha2, theta1, theta2

# Terrain parameters
TERRAIN_SIZE = (8.0, 8.0)
TERRAIN_GRANULARITY = 0.5
RUGGEDNESS_RANGE = np.arange(0.0, 0.5, 0.1)
BOWLNESS_RANGE = np.arange(0.0, 2.5, 0.5)

ROBOT_INITIAL_Z_OFFSET = 0.5
