from typing import Tuple, List
from revolve2.core.modular_robot import ActiveHinge, Body, Brick
import math


def make_body_1() -> Tuple[Body, List[int]]:
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.attachment = Brick(0.0)

    body.finalize()

    dof_map = {body.core.left.id: 0, body.core.left.attachment.id: 1}

    return body, dof_map


def make_body_2() -> Tuple[Body, List[int]]:
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment.attachment = Brick(0.0)

    body.finalize()

    dof_map = {
        body.core.left.id: 0,
        body.core.left.attachment.id: 1,
        body.core.right.id: 2,
        body.core.right.attachment.id: 3,
    }

    return body, dof_map


def make_bodies() -> Tuple[List[Body], List[List[int]]]:
    """
    :returns: Bodies and corresponding maps from active hinge id to dof index
    """

    body1, dof_map1 = make_body_1()
    body2, dof_map2 = make_body_2()

    return [body1, body2], [dof_map1, dof_map2]
