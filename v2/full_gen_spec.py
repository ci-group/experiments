from audioop import cross
import logging

from de_multi_body_optimizer import DEMultiBodyOptimizer

from revolve2.core.database import open_async_database_sqlite
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from typing import List
from revolve2.core.modular_robot import Body
from revolve2.core.database.std import Rng
import numpy as np


async def run_full_gen_spec(
    database_name: str,
    bodies: List[Body],
    dof_maps: List[List[int]],
    cpg_network_structure: CpgNetworkStructure,
    headless: bool,
    rng_seed: int,
    num_evaluations: int,
    population_size: int,
    crossover_probability: float,
    differential_weight: float,
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

    await DEMultiBodyOptimizer().run(
        rng=rng,
        database=database,
        robot_bodies=bodies,
        dof_maps=dof_maps,
        cpg_network_structure=cpg_network_structure,
        headless=headless,
        num_evaluations=num_evaluations,
        population_size=population_size,
        crossover_probability=crossover_probability,
        differential_weight=differential_weight,
    )

    logging.info(f"Finished optimizing.")
