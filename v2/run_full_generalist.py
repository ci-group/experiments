from full_gen_spec import run_full_gen_spec
from bodies import make_bodies, make_cpg_network_structure


async def main() -> None:
    # await run_all_full_generalist_runs()  # For the actual experiments
    await run_full_generalist(
        f"dbg_full_generalist", True, 0, 0.05, 0.01
    )  # For debugging only


async def run_all_full_generalist_runs() -> None:
    NUM_RUNS = 20
    for (sigma, learning_rate) in [(0.5, 0.1), (0.05, 0.01)]:
        for i in range(NUM_RUNS):
            await run_full_generalist(
                f"dbs/full_generalist_s{sigma}l{learning_rate}_run{i}",
                True,
                i,
                sigma=sigma,
                learning_rate=learning_rate,
            )


async def run_full_generalist(
    database_name: str,
    headless: bool,
    rng_seed: int,
    sigma: float,
    learning_rate: float,
) -> None:
    NUM_EVALUATIONS = 10000

    bodies, dof_maps = make_bodies()
    cpg_network_structure = make_cpg_network_structure()

    await run_full_gen_spec(
        database_name,
        bodies,
        dof_maps,
        cpg_network_structure,
        headless,
        rng_seed,
        NUM_EVALUATIONS,
        sigma=sigma,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
