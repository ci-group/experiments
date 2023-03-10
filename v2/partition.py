from graph import Graph, Node
from environment import Environment
from typing import List
import math
import numpy as np
from collections import deque
import random
import csv
from pathlib import Path


def dist(env1: Environment, env2: Environment) -> float:
    return math.sqrt(
        (env1.name.body_num - env2.name.body_num) ** 2
        + (env1.name.ruggedness_num - env2.name.ruggedness_num) ** 2
        + (env1.name.bowlness_num - env2.name.bowlness_num) ** 2
    )


def partition(
    graph: Graph, envs: List[Environment], num_partitions: int
) -> List[Graph]:
    if num_partitions == 125:
        subgraphs = [[graph.nodes[i]] for i in range(num_partitions)]
    elif num_partitions == 1:
        return [graph]
    elif num_partitions == 25:
        subgraphs: List[List[Node]] = [[]] * 25
        with Path(__file__).with_name("partition_psize5.txt").open("r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                subgraphs[int(row[0])].append(graph.nodes[int(row[1])])
    elif num_partitions == 5:
        subgraphs: List[List[Node]] = [[]] * 5
        with Path(__file__).with_name("partition_psize25.txt").open("r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                subgraphs[int(row[0])].append(graph.nodes[int(row[1])])
    else:
        raise NotImplementedError()

    for subgraph in subgraphs:
        for node in subgraph:
            node.neighbor = [
                neighbor for neighbor in node.neighbours if neighbor in subgraph
            ]
    return [Graph(subgraph) for subgraph in subgraphs]


# def find_start_points(
#     graph: Graph, envs: List[Environment], num_points: int, rng: random.Random
# ) -> List[Node]:
#     points = [
#         graph.nodes[rng.randint(0, len(graph.nodes) - 1)] for _ in range(num_points)
#     ]
#     for _ in range(500):
#         for i, point in enumerate(points):
#             neighbor_scores = [
#                 min(
#                     [
#                         dist(envs[neighbor.index], envs[other_point.index])
#                         for j, other_point in enumerate(points)
#                         if i != j
#                     ]
#                 )
#                 for neighbor in point.neighbours
#             ]
#             best = np.argmax(neighbor_scores)
#             points[i] = point.neighbours[best]
#     return points


# def partition_old(
#     graph: Graph, envs: List[Environment], num_partitions: int
# ) -> List[Graph]:
#     rng = random.Random()
#     rng.seed(100)

#     subgraphs: List[List[Node]]

#     if num_partitions == 125:
#         subgraphs = [[graph.nodes[i]] for i in range(num_partitions)]
#     elif num_partitions == 1:
#         return [graph]
#     else:
#         partition_centers = find_start_points(graph, envs, num_partitions, rng)

#         opens: List[deque[Node]] = [deque() for _ in range(num_partitions)]
#         taken: List[Node] = []
#         subgraphs = [[] for _ in range(num_partitions)]

#         for i in range(num_partitions):
#             taken.append(partition_centers[i])
#             subgraphs[i].append(partition_centers[i])
#             for neighbor in partition_centers[i].neighbours:
#                 opens[i].append(neighbor)

#         while len(taken) < len(graph.nodes):
#             for i in range(num_partitions):
#                 item = None
#                 while len(opens[i]) != 0:
#                     get = opens[i].pop()
#                     if get not in taken:
#                         item = get
#                         break

#                 if item is not None:
#                     subgraphs[i].append(item)
#                     taken.append(item)
#                     for neighbor in item.neighbours:
#                         if neighbor not in taken:
#                             opens[i].append(neighbor)

#     for subgraph in subgraphs:
#         for node in subgraph:
#             node.neighbor = [
#                 neighbor for neighbor in node.neighbours if neighbor in subgraph
#             ]
#     return [Graph(subgraph) for subgraph in subgraphs]
