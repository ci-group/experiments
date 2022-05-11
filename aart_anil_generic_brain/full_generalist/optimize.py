import logging
import math
from random import Random, sample

from optimizer import Optimizer

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.modular_robot import ActiveHinge, Body, Brick
from revolve2.core.optimization import ProcessIdGen
from typing import Tuple, List
from revolve2.actor_controllers.cpg import CpgNetworkStructure, CpgPair


def make_body() -> Tuple[Body, List[int]]:
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.attachment = Brick(0.0)

    body.finalize()

    dof_map = {body.core.left.id: 0, body.core.left.attachment.id: 1}

    return body, dof_map


async def main() -> None:
    POPULATION_SIZE = 10
    SIGMA = 0.1
    LEARNING_RATE = 0.05
    NUM_GENERATIONS = 3

    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # random number generator
    rng = Random()
    rng.seed(0)

    # database
    database = open_async_database_sqlite("./database")

    # process id generator
    process_id_gen = ProcessIdGen()

    body, dof_map = make_body()

    robot_bodies = [body]
    dof_maps = [dof_map]

    cpgs = CpgNetworkStructure.make_cpgs(2)
    cpg_network_structure = CpgNetworkStructure(cpgs, set([CpgPair(cpgs[0], cpgs[1])]))

    process_id = process_id_gen.gen()
    maybe_optimizer = await Optimizer.from_database(
        database=database,
        process_id=process_id,
        process_id_gen=process_id_gen,
        rng=rng,
        robot_bodies=robot_bodies,
        dof_maps=dof_maps,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_generations=NUM_GENERATIONS,
        cpg_network_structure=cpg_network_structure,
    )
    if maybe_optimizer is not None:
        logging.info(
            f"Recovered. Last finished generation: {maybe_optimizer.generation_number}."
        )
        optimizer = maybe_optimizer
    else:
        logging.info(f"No recovery data found. Starting at generation 0.")
        optimizer = await Optimizer.new(
            database,
            process_id,
            process_id_gen,
            rng,
            POPULATION_SIZE,
            SIGMA,
            LEARNING_RATE,
            robot_bodies=robot_bodies,
            dof_maps=dof_maps,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            cpg_network_structure=cpg_network_structure,
        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
