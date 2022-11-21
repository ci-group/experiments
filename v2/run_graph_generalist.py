from bodies import make_cpg_network_structure, make_bodies
import logging
from revolve2.core.database import open_async_database_sqlite
from graph_optimizer import GraphOptimizer
from graph import Graph, Node
from revolve2.core.database.std import Rng
from typing import List, Tuple
from bodies import (
    make_cpg_network_structure,
)
from sqlalchemy.ext.asyncio.session import AsyncSession
from environment import Environment
import numpy as np
from graph import Graph
from experiment_settings import (
    GRAPH_PARAMS,
    NUM_EVALUATIONS,
    RUGGEDNESS_RANGE,
    BOWLNESS_RANGE,
    TERRAIN_SIZE,
    TERRAIN_GRANULARITY,
)
import argparse
from terrain_generator import terrain_generator
from environment_name import EnvironmentName

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
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=int)
    args = parser.parse_args()

    for standard_deviation in GRAPH_PARAMS:
        await run_graph_generalist(
            database_name=f"dbs/graph_s{standard_deviation}_run{args.run}",
            headless=True,
            rng_seed=SEED_BASE + len(GRAPH_PARAMS * args.run),
            standard_deviation=standard_deviation,
        )


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
                        bodies[body_i],
                        dof_maps[body_i],
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
    database = open_async_database_sqlite(database_name, create=True)

    logging.info("Generating terrains..")

    # graph description
    graph, environments = make_graph()

    cpg_network_structure = make_cpg_network_structure()

    logging.info("Generating done.")

    # save environment names. bit of a hack to not double save
    async with database.begin() as conn:
        await EnvironmentName.prepare_db(conn)

    async with AsyncSession(database) as ses:
        async with ses.begin():
            if (await EnvironmentName.from_db(ses, 1)) is None:
                await EnvironmentName.to_db_multiple(
                    ses, [env.name for env in environments]
                )

    logging.info("Starting optimization process..")

    await GraphOptimizer().run(
        rng=rng,
        database=database,
        environments=environments,
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
