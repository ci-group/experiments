from typing import Tuple, List
from revolve2.core.modular_robot import ActiveHinge, Body, Brick
import math
from revolve2.actor_controllers.cpg import CpgNetworkStructure, CpgPair


def make_body_1() -> Tuple[Body, List[int]]:
    # supergecko
    body = Body()
    body.core.left = ActiveHinge(math.pi / 2.0)  # id 0
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)  # id 1
    body.core.left.attachment.attachment = Brick(0.0)

    body.core.right = ActiveHinge(math.pi / 2.0)  # id 2
    body.core.right.attachment = ActiveHinge(math.pi / 2.0)  # id 3
    body.core.right.attachment.attachment = Brick(0.0)

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment = Brick(math.pi / 2.0)

    body.core.back.attachment.attachment.left = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment.attachment = Brick(0.0)

    body.core.back.attachment.attachment.right = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.right.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.right.attachment.attachment = Brick(0.0)

    body.finalize()

    dof_map = {
        body.core.left.id: 0,
        body.core.left.attachment.id: 1,
        body.core.right.id: 2,
        body.core.right.attachment.id: 3,
        body.core.back.id: 4,
        body.core.back.attachment.id: 5,
        body.core.back.attachment.attachment.left.id: 6,
        body.core.back.attachment.attachment.left.attachment.id: 7,
        body.core.back.attachment.attachment.right.id: 8,
        body.core.back.attachment.attachment.right.attachment.id: 9,
    }

    return body, dof_map







def make_body_2() -> Tuple[Body, List[int]]:
    # supergecko front left missing
    body = Body()

    body.core.right = ActiveHinge(math.pi / 2.0)  # id 2
    body.core.right.attachment = ActiveHinge(math.pi / 2.0)  # id 3
    body.core.right.attachment.attachment = Brick(0.0)

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment = Brick(math.pi / 2.0)

    body.core.back.attachment.attachment.left = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment.attachment = Brick(0.0)

    body.core.back.attachment.attachment.right = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.right.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.right.attachment.attachment = Brick(0.0)

    body.finalize()

    dof_map = {
        body.core.right.id: 2,
        body.core.right.attachment.id: 3,
        body.core.back.id: 4,
        body.core.back.attachment.id: 5,
        body.core.back.attachment.attachment.left.id: 6,
        body.core.back.attachment.attachment.left.attachment.id: 7,
        body.core.back.attachment.attachment.right.id: 8,
        body.core.back.attachment.attachment.right.attachment.id: 9,
    }

    return body, dof_map





def make_body_3() -> Tuple[Body, List[int]]:
    # supergecko back right missing
    body = Body()
    body.core.left = ActiveHinge(math.pi / 2.0)  # id 0
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)  # id 1
    body.core.left.attachment.attachment = Brick(0.0)

    body.core.right = ActiveHinge(math.pi / 2.0)  # id 2
    body.core.right.attachment = ActiveHinge(math.pi / 2.0)  # id 3
    body.core.right.attachment.attachment = Brick(0.0)

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment = Brick(math.pi / 2.0)

    body.core.back.attachment.attachment.left = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment.attachment = Brick(0.0)


    body.finalize()

    dof_map = {
        body.core.left.id: 0,
        body.core.left.attachment.id: 1,
        body.core.right.id: 2,
        body.core.right.attachment.id: 3,
        body.core.back.id: 4,
        body.core.back.attachment.id: 5,
        body.core.back.attachment.attachment.left.id: 6,
        body.core.back.attachment.attachment.left.attachment.id: 7,
    }

    return body, dof_map





def make_body_4() -> Tuple[Body, List[int]]:
    # supergecko front both missing
    body = Body()
    

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment = Brick(math.pi / 2.0)

    body.core.back.attachment.attachment.left = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.left.attachment.attachment = Brick(0.0)

    body.core.back.attachment.attachment.right = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.right.attachment = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.attachment.right.attachment.attachment = Brick(0.0)

    body.finalize()

    dof_map = {
    
        body.core.back.id: 4,
        body.core.back.attachment.id: 5,
        body.core.back.attachment.attachment.left.id: 6,
        body.core.back.attachment.attachment.left.attachment.id: 7,
        body.core.back.attachment.attachment.right.id: 8,
        body.core.back.attachment.attachment.right.attachment.id: 9,
    }

    return body, dof_map






def make_body_5() -> Tuple[Body, List[int]]:
    # geodude
    body = Body()
    body.core.left = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.right = ActiveHinge(math.pi / 2.0)
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
    body3, dof_map3 = make_body_3()
    body4, dof_map4 = make_body_4()
    body5, dof_map5 = make_body_5()

    return [body1, body2, body3, body4, body5], [dof_map1, dof_map2, dof_map3, dof_map4, dof_map5]


def make_cpg_network_structure() -> CpgNetworkStructure:
    cpgs = CpgNetworkStructure.make_cpgs(10)
    cpg_network_structure = CpgNetworkStructure(
        cpgs,
        set(
            [
                CpgPair(cpgs[0], cpgs[1]),
                CpgPair(cpgs[2], cpgs[3]),
                CpgPair(cpgs[0], cpgs[2]),
                CpgPair(cpgs[0], cpgs[4]),
                CpgPair(cpgs[2], cpgs[4]),
                CpgPair(cpgs[4], cpgs[5]),
                CpgPair(cpgs[6], cpgs[7]),
                CpgPair(cpgs[8], cpgs[9]),
                CpgPair(cpgs[6], cpgs[8]),
                CpgPair(cpgs[5], cpgs[6]),
                CpgPair(cpgs[5], cpgs[8]),
            ]
        ),
    )
    return cpg_network_structure
