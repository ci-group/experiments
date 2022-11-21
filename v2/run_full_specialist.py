from full_gen_spec import run_full_gen_spec
from bodies import make_bodies, make_cpg_network_structure
from typing import List
from experiment_settings import (
    DE_PARAMS,
    NUM_EVALUATIONS,
    NUM_RUNS,
    TERRAIN_SIZE,
    TERRAIN_GRANULARITY,
    RUGGEDNESS_RANGE,
    BOWLNESS_RANGE,
)
from environment import Environment
import itertools
from terrain_generator import terrain_generator
from environment_name import EnvironmentName

SEED_BASE = 196783254


async def main() -> None:
    await run_all_full_specialist_runs()  # For the actual experiments
    # await dbg_run_full_specialist()


async def dbg_run_full_specialist() -> None:
    bodies, dof_maps = make_bodies()
    body_num = 2
    ruggedness_num = 2
    bowlness_num = 2
    await run_full_specialist(
        database_name=f"dbg_full_specialist_body{body_num}",
        headless=True,
        rng_seed=0,
        environment=Environment(
            bodies[body_num],
            dof_maps[body_num],
            terrain_generator(
                TERRAIN_SIZE,
                RUGGEDNESS_RANGE[ruggedness_num],
                BOWLNESS_RANGE[bowlness_num],
                TERRAIN_GRANULARITY,
            ),
        ),
        population_size=100,
        crossover_probability=0.9,
        differential_weight=0.8,
    )


async def run_all_full_specialist_runs() -> None:
    bodies, dof_maps = make_bodies()

    environments = [
        Environment(
            body,
            dof_map,
            terrain_generator(TERRAIN_SIZE, ruggedness, bowlness, TERRAIN_GRANULARITY),
            EnvironmentName(body_i, ruggedness_i, bowlness_i),
        )
        for (body_i, (body, dof_map)), (ruggedness_i, ruggedness), (
            bowlness_i,
            bowlness,
        ) in itertools.product(
            enumerate(zip(bodies, dof_maps)),
            enumerate(RUGGEDNESS_RANGE),
            enumerate(BOWLNESS_RANGE),
        )
    ]

    seed = SEED_BASE

    for (population_size, crossover_probability, differential_weight) in DE_PARAMS:
        for environment in environments:
            for i_run in range(NUM_RUNS):
                await run_full_specialist(
                    database_name=f"dbs/full_specialist_p{population_size}_cr{crossover_probability}_f{differential_weight}_body{environment.name.body_num}_ruggedness{environment.name.ruggedness_num}_bowlness{environment.name.bowlness_num}_run{i_run}",
                    headless=True,
                    rng_seed=seed,
                    environment=environment,
                    population_size=population_size,
                    crossover_probability=crossover_probability,
                    differential_weight=differential_weight,
                )
                seed += 1


async def run_full_specialist(
    database_name: str,
    headless: bool,
    rng_seed: int,
    environment: Environment,
    population_size: float,
    crossover_probability: float,
    differential_weight: float,
) -> None:
    cpg_network_structure = make_cpg_network_structure()
    num_evaluations = NUM_EVALUATIONS / len(
        make_bodies()[0] * len(RUGGEDNESS_RANGE) * len(BOWLNESS_RANGE)
    )

    await run_full_gen_spec(
        database_name=database_name,
        environments=[environment],
        cpg_network_structure=cpg_network_structure,
        headless=headless,
        rng_seed=rng_seed,
        num_evaluations=num_evaluations,
        population_size=population_size,
        crossover_probability=crossover_probability,
        differential_weight=differential_weight,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
