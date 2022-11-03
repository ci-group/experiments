from full_gen_spec import run_full_gen_spec
from bodies import make_bodies, make_cpg_network_structure
from revolve2.core.modular_robot import Body
from typing import List
from experiment_settings import DE_PARAMS, NUM_EVALUATIONS, NUM_RUNS

SEED_BASE = 196783254


async def main() -> None:
    await run_all_full_specialist_runs()  # For the actual experiments
    # await dbg_run_full_specialist()


async def dbg_run_full_specialist() -> None:
    bodies, dof_maps = make_bodies()
    body_num = 2
    await run_full_specialist(
        database_name=f"dbg_full_specialist_body{body_num}",
        headless=True,
        rng_seed=0,
        body=bodies[body_num],
        dof_map=dof_maps[body_num],
        population_size=100,
        crossover_probability=0.9,
        differential_weight=0.8,
    )


async def run_all_full_specialist_runs() -> None:
    bodies, dof_maps = make_bodies()

    seed = SEED_BASE

    for (population_size, crossover_probability, differential_weight) in DE_PARAMS:
        for i_body, (body, dof_map) in enumerate(zip(bodies, dof_maps)):
            for i_run in range(NUM_RUNS):
                await run_full_specialist(
                    database_name=f"dbs/full_specialist_p{population_size}_cr{crossover_probability}_f{differential_weight}_body{i_body}_run{i_run}",
                    headless=True,
                    rng_seed=seed,
                    body=body,
                    dof_map=dof_map,
                    population_size=population_size,
                    crossover_probability=crossover_probability,
                    differential_weight=differential_weight,
                )
                seed += 1


async def run_full_specialist(
    database_name: str,
    headless: bool,
    rng_seed: int,
    body: Body,
    dof_map: List[int],
    population_size: float,
    crossover_probability: float,
    differential_weight: float,
) -> None:
    cpg_network_structure = make_cpg_network_structure()
    num_evaluations = NUM_EVALUATIONS / len(make_bodies()[0])

    await run_full_gen_spec(
        database_name=database_name,
        bodies=[body],
        dof_maps=[dof_map],
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
