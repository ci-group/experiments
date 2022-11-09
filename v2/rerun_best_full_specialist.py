from bodies import make_bodies, make_cpg_network_structure

from rerun_best_generic import rerun_best
import argparse


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to fetch brain parameters from.",
    )
    parser.add_argument(
        "body",
        type=int,
        help="Number of the body to simulate.",
    )
    args = parser.parse_args()

    await rerun_best(
        database_name=args.database,
        body=args.body,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
