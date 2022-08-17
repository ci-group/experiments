import logging
from bodies import make_bodies
from random import Random, sample

from openaies_multi_body_optimizer import OpenaiESMultiBodyOptimizer

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from typing import List
from revolve2.core.modular_robot import Body


async def run_full_gen_spec(
    database_name: str,
    bodies: List[Body],
    dof_maps: List[List[int]],
    cpg_network_structure: CpgNetworkStructure,
    headless: bool,
    rng_seed: int,
    num_evaluations: int,
    sigma: float,
    learning_rate: float,
) -> None:
    POPULATION_SIZE = 100

    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 60

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

    process_id = process_id_gen.gen()
    maybe_optimizer = await OpenaiESMultiBodyOptimizer.from_database(
        database=database,
        process_id=process_id,
        process_id_gen=process_id_gen,
        rng=rng,
        robot_bodies=bodies,
        dof_maps=dof_maps,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_evaluations=num_evaluations,
        cpg_network_structure=cpg_network_structure,
        headless=headless,
    )
    if maybe_optimizer is not None:
        logging.info(
            f"Recovered. Last finished generation: {maybe_optimizer.generation_number}."
        )
        optimizer = maybe_optimizer
    else:
        logging.info(f"No recovery data found. Starting at generation 0.")
        optimizer = await OpenaiESMultiBodyOptimizer.new(
            database,
            process_id,
            process_id_gen,
            rng,
            population_size=POPULATION_SIZE,
            sigma=sigma,
            learning_rate=learning_rate,
            robot_bodies=bodies,
            dof_maps=dof_maps,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_evaluations=num_evaluations,
            cpg_network_structure=cpg_network_structure,
            headless=headless,
        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info(f"Finished optimizing.")
