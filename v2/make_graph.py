from graph import Graph, Node
from typing import List, Tuple
from environment import Environment
from bodies import make_bodies
from experiment_settings import (
    RUGGEDNESS_RANGE,
    BOWLNESS_RANGE,
    TERRAIN_SIZE,
    TERRAIN_GRANULARITY,
    RANDOM_GRAPH_CONNECT_PROB,
)
from terrain_generator import terrain_generator
from environment_name import EnvironmentName
import numpy as np


def make_graph() -> Tuple[Graph, List[Environment]]:
    bodies, dof_maps = make_bodies()

    nodes: List[Node] = []
    envs: List[Environment] = []

    for ruggedness_i, ruggedness in enumerate(RUGGEDNESS_RANGE):
        for bowlness_i, bowlness in enumerate(BOWLNESS_RANGE):
            terrain = terrain_generator(
                size=TERRAIN_SIZE,
                ruggedness=ruggedness,
                bowlness=bowlness,
                granularity_multiplier=TERRAIN_GRANULARITY,
            )
            for body_i, (body, dof_map) in enumerate(zip(bodies, dof_maps)):
                node = Node(len(nodes))
                nodes.append(node)
                envs.append(
                    Environment(
                        body,
                        dof_map,
                        terrain,
                        EnvironmentName(body_i, ruggedness_i, bowlness_i),
                    )
                )
                nodes[-1].env = envs[-1]

    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:
                continue

            env1 = envs[node1.index]
            env2 = envs[node2.index]

            if (
                abs(env1.name.body_num - env2.name.body_num) <= 1
                and abs(env1.name.ruggedness_num - env2.name.ruggedness_num) <= 1
                and abs(env1.name.bowlness_num - env2.name.bowlness_num) <= 1
            ):
                node1.neighbours.append(node2)

    return Graph(nodes), envs


def make_random_graph(rng: np.random.Generator) -> Tuple[Graph, List[Environment]]:
    bodies, dof_maps = make_bodies()

    nodes: List[Node] = []
    envs: List[Environment] = []

    for ruggedness_i, ruggedness in enumerate(RUGGEDNESS_RANGE):
        for bowlness_i, bowlness in enumerate(BOWLNESS_RANGE):
            terrain = terrain_generator(
                size=TERRAIN_SIZE,
                ruggedness=ruggedness,
                bowlness=bowlness,
                granularity_multiplier=TERRAIN_GRANULARITY,
            )
            for body_i, (body, dof_map) in enumerate(zip(bodies, dof_maps)):
                node = Node(len(nodes))
                nodes.append(node)
                envs.append(
                    Environment(
                        body,
                        dof_map,
                        terrain,
                        EnvironmentName(body_i, ruggedness_i, bowlness_i),
                    )
                )
                nodes[-1].env = envs[-1]

    for i, node1 in enumerate(nodes):
        for node2 in nodes[i + 1 :]:
            if rng.random() < RANDOM_GRAPH_CONNECT_PROB:
                node1.neighbours.append(node2)
                node2.neighbours.append(node1)

    return Graph(nodes), envs
