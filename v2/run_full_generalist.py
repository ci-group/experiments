from full_gen_spec import run_full_gen_spec
from bodies import make_bodies, make_cpg_network_structure
from experiment_settings import NUM_EVALUATIONS, NUM_RUNS, DE_PARAMS

SEED_BASE = 23678400


async def main() -> None:
    await run_all_full_generalist_runs()  # For the actual experiments
    # await dbg_run_full_generalist()


async def dbg_run_full_generalist() -> None:
    await run_full_generalist(
        database_name=f"dbg_full_generalist",
        headless=True,
        rng_seed=0,
        population_size=100,
        crossover_probability=0.9,
        differential_weight=0.8,
    )


async def run_all_full_generalist_runs() -> None:
    seed = SEED_BASE

    for (population_size, crossover_probability, differential_weight) in DE_PARAMS:
        for i in range(NUM_RUNS):
            await run_full_generalist(
                database_name=f"dbs/full_generalist_p{population_size}_cr{crossover_probability}_f{differential_weight}_run{i}",
                headless=True,
                rng_seed=seed,
                population_size=population_size,
                crossover_probability=crossover_probability,
                differential_weight=differential_weight,
            )
            seed += 1


async def run_full_generalist(
    database_name: str,
    headless: bool,
    rng_seed: int,
    population_size: float,
    crossover_probability: float,
    differential_weight: float,
) -> None:
    bodies, dof_maps = make_bodies()
    cpg_network_structure = make_cpg_network_structure()

    await run_full_gen_spec(
        database_name=database_name,
        bodies=bodies,
        dof_maps=dof_maps,
        cpg_network_structure=cpg_network_structure,
        headless=headless,
        rng_seed=rng_seed,
        num_evaluations=NUM_EVALUATIONS,
        population_size=population_size,
        crossover_probability=crossover_probability,
        differential_weight=differential_weight,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
