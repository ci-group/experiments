from bodies import make_cpg_network_structure, make_bodies
import logging
from random import Random
from revolve2.core.database import open_async_database_sqlite
from graph_optimizer import GraphOptimizer
from graph import Graph, Node
from revolve2.core.database.std import Rng
from typing import List
from bodies import (
    make_body_1,
    make_body_2,
    make_body_3,
    make_body_4,
    make_body_5,
    make_cpg_network_structure,
)
from revolve2.actor_controllers.cpg import CpgNetworkStructure
import numpy as np
from graph import Graph
from experiment_settings import GRAPH_PARAMS, NUM_RUNS, NUM_EVALUATIONS

SEED_BASE = 732091019


async def main() -> None:
    # await run_all_graph_generalist_runs()  # For the actual experiments
    await dbg_run_graph()


async def dbg_run_graph() -> None:
    await run_graph_generalist(
        database_name=f"dbg_graph",
        headless=True,
        rng_seed=0,
        standard_deviation=0.2,
    )


async def run_all_graph_generalist_runs() -> None:
    seed = SEED_BASE

    for standard_deviation in GRAPH_PARAMS:
        for i in range(NUM_RUNS):
            await run_graph_generalist(
                database_name=f"dbs/graph_s{standard_deviation}_run{i}",
                headless=True,
                rng_seed=seed,
                standard_deviation=standard_deviation,
            )
            seed += 1


def make_graph() -> Graph:
    nodes = [Node(i) for i in range(5)]

    nodes[0].neighbours.append(nodes[1])

    nodes[1].neighbours.append(nodes[0])
    nodes[1].neighbours.append(nodes[2])

    nodes[2].neighbours.append(nodes[1])
    nodes[2].neighbours.append(nodes[3])

    nodes[3].neighbours.append(nodes[2])
    nodes[3].neighbours.append(nodes[4])

    nodes[4].neighbours.append(nodes[3])

    return Graph([nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]])


async def run_graph_generalist(
    database_name: str, headless: bool, rng_seed: int, standard_deviation: float
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # random number generator
    rng = Rng(np.random.Generator(np.random.PCG64(rng_seed)))

    logging.info(f"Opening database: {database_name}")

    # database
    database = open_async_database_sqlite(database_name)

    logging.info("Starting optimization process..")

    # graph description
    bodies, dof_maps = make_bodies()
    cpg_network_structure = make_cpg_network_structure()
    graph = make_graph()

    logging.info("Starting optimization process..")

    await GraphOptimizer().run(
        rng=rng,
        database=database,
        robot_bodies=bodies,
        dof_maps=dof_maps,
        graph=graph,
        cpg_network_structure=cpg_network_structure,
        headless=headless,
        num_evaluations=NUM_EVALUATIONS,
        standard_deviation=standard_deviation,
    )

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
