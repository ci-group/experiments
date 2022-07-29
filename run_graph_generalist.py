from full_gen_spec import run_full_gen_spec
from bodies import make_bodies, make_cpg_network_structure
import logging
from random import Random
from revolve2.core.database import open_async_database_sqlite
from graph_generalist_optimizer import GraphGeneralistOptimizer, GraphNode, Environment
from revolve2.core.optimization import ProcessIdGen
from typing import List
from bodies import make_body_1, make_body_2, make_cpg_network_structure
from revolve2.actor_controllers.cpg import CpgNetworkStructure
import numpy as np


async def main() -> None:
    # await run_all_graph_generalist_runs() # For the actual experiments
    await run_graph_generalist(f"dbg_graph_generalist", True, 0)  # For debugging only


async def run_all_graph_generalist_runs() -> None:
    NUM_RUNS = 20
    for i in range(NUM_RUNS):
        await run_graph_generalist(f"full_generalist_run{i}", True, i)


def make_graph_nodes(
    rng: Random, cpg_network_structure: CpgNetworkStructure
) -> List[GraphNode]:
    nprng = np.random.Generator(
        np.random.PCG64(rng.randint(0, 2**63))
    )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
    genotype1 = nprng.uniform(
        size=cpg_network_structure.num_params,
        low=0,
        high=1.0,
    )
    genotype2 = nprng.uniform(
        size=cpg_network_structure.num_params,
        low=0,
        high=1.0,
    )

    body1, dof_map1 = make_body_1()
    body2, dof_map2 = make_body_2()
    env1 = Environment(body1, dof_map1)
    env2 = Environment(body2, dof_map2)
    edges1 = []
    edges2 = []
    node1 = GraphNode(env1, genotype1, edges1, None)
    node2 = GraphNode(env2, genotype2, edges2, None)
    edges1.append(node2)
    edges2.append(node1)

    return [node1, node2]


async def run_graph_generalist(
    database_name: str,
    headless: bool,
    rng_seed: int,
) -> None:
    NUM_GENERATIONS = 100

    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 10

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # random number generator
    rng = Random()
    rng.seed(rng_seed)

    logging.info(f"Opening database: {database_name}")

    # database
    database = open_async_database_sqlite(database_name)

    # process id generator
    process_id_gen = ProcessIdGen()

    # graph description
    cpg_network_structure = make_cpg_network_structure()
    graph_nodes = make_graph_nodes(rng, cpg_network_structure)

    process_id = process_id_gen.gen()
    maybe_optimizer = await GraphGeneralistOptimizer.from_database(
        database=database,
        process_id=process_id,
        process_id_gen=process_id_gen,
    )
    if maybe_optimizer is not None:
        logging.info(
            f"Recovered. Last finished generation: {maybe_optimizer.generation_number}."
        )
        optimizer = maybe_optimizer
    else:
        logging.info(f"No recovery data found. Starting at generation 0.")
        optimizer = await GraphGeneralistOptimizer.new(
            database=database,
            process_id=process_id,
            process_id_gen=process_id_gen,
            rng=rng,
            num_generations=NUM_GENERATIONS,
            graph_nodes=graph_nodes,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            cpg_network_structure=cpg_network_structure,
            headless=headless,
        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
