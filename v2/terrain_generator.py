from revolve2.core.physics import Terrain
from pyrr import Vector3, Quaternion
from revolve2.core.physics.running import geometry
from typing import Tuple, List
from noise import pnoise2
import math


def terrain_generator(
    size: Tuple[float, float],
    ruggedness: float,
    bowlness: float,
    granularity_multiplier: float = 1.0,
) -> Terrain:
    NUM_EDGES = 100

    num_edges = (
        int(NUM_EDGES * size[0] * granularity_multiplier),
        int(NUM_EDGES * size[1] * granularity_multiplier),
    )

    rugged = rugged_heightmap(
        size=size,
        num_edges=num_edges,
        density=1.5,
    )
    bowl = bowl_heighmap(num_edges=num_edges)

    heightmap = [
        [
            0.0
            if ruggedness + bowlness == 0.0
            else (yrugged * ruggedness + ybowl * bowlness) / (ruggedness + bowlness)
            for yrugged, ybowl in zip(xrugged, xbowl)
        ]
        for xrugged, xbowl in zip(rugged, bowl)
    ]
    max_height = ruggedness + bowlness

    return Terrain(
        static_geometry=[
            geometry.Heightmap(
                position=Vector3(),
                orientation=Quaternion(),
                size=Vector3([size[0], size[1], max_height]),
                base_thickness=0.1 + ruggedness,
                heights=heightmap,
            )
        ]
    )


def rugged_heightmap(
    size: Tuple[float, float],
    num_edges: Tuple[int, int],
    density: float = 1.0,
) -> List[List[float]]:
    # TODO the maximum height is not really 1.0.
    OCTAVE = 10
    C1 = 4.0  # arbitrary constant to get nice noise

    return [
        [
            pnoise2(
                x / num_edges[0] * C1 * size[0] * density,
                y / num_edges[1] * C1 * size[1] * density,
                OCTAVE,
            )
            for y in range(num_edges[1])
        ]
        for x in range(num_edges[0])
    ]


def bowl_heighmap(
    num_edges: Tuple[int, int],
) -> List[List[float]]:
    return [
        [
            (x / num_edges[0] * 2.0 - 1.0) ** 2 + (y / num_edges[1] * 2.0 - 1.0) ** 2
            if math.sqrt(
                (x / num_edges[0] * 2.0 - 1.0) ** 2
                + (y / num_edges[1] * 2.0 - 1.0) ** 2
            )
            <= 1.0
            else 0.0
            for y in range(num_edges[1])
        ]
        for x in range(num_edges[0])
    ]
