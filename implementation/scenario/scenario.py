import math
import random

import numpy as np

from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.const import JointMode
from pyrep.objects.shape import Shape

from .dobot import Dobot

from typing import List


class Scenario:
    def __init__(self, num_cubes=8):
        self.pr = PyRep()
        self.num_cubes = num_cubes

    def reset(self):
        if self.pr.running:
            self.pr.stop()
            self.pr.shutdown()
        self.pr.launch('scenario.ttt', headless=True)

        self.arm = Dobot()
        self.cubes = []
        for _ in range(self.num_cubes):
            angle = 2 * math.pi * (random.random() * 0.5 + 0.5)
            radius = random.random() * 0.1 + 0.15
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            self.cubes.append(Shape.create(
                type=PrimitiveShape.CUBOID,
                position=[x, y, 0.0025],
                size=[0.05, 0.05, 0.05],
                color=[random.random() for _ in range(3)],
            ))

        self.pr.start()

        return self._create_observation()

    def step(self, action: List[float]):
        joint_targets = action[:-1]
        for joint, target in zip(self.arm.joints, joint_targets):
            joint.set_joint_target_velocity(target)

        suction_cup_target = action[-1]
        if suction_cup_target > 0.0:
            # Try to grasp each cube in the scene.
            if len(self.arm.suction_cup.get_grasped_objects()) == 0:
                for cube in self.cubes:
                    if self.arm.suction_cup.grasp(cube):
                        break
        else:
            self.arm.suction_cup.release()

        self.pr.step()

        reward = self._get_reward()
        done = self.pr.get_simulation_timestep() > 16

        return self._create_observation(), reward, done

    def close(self):
        self.pr.shutdown()

    def _create_observation(self):
        return [
            *self._get_joint_positions(),
            *self._get_cube_positions(),
        ]

    def _get_joint_positions(self):
        return [joint.get_joint_position() for joint in self.arm.joints]

    def _get_cube_positions(self):
        result = []
        for cube in self.cubes:
            result.extend(cube.get_position())
        return result

    def _get_reward(self):
        grasped_objects = len(self.arm.suction_cup.get_grasped_objects())
        proximity_bonus = 0.0
        sensor_pos = np.array(self.arm.suction_cup_sensor.get_position())
        for cube in self.cubes:
            cube_pos = np.array(cube.get_position())
            distance = np.linalg.norm(cube_pos - sensor_pos)
            proximity_bonus += 1.0 / ((distance * 8.0 + 1.0) ** 2)

        return grasped_objects + proximity_bonus * 0.1
