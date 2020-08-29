import math
import random
from os.path import dirname, join, abspath

import numpy as np

import gym

from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.const import JointMode
from pyrep.objects.shape import Shape

from scenario.dobot import Dobot


SCENE_FILE = join(dirname(abspath(__file__)), 'scenario.ttt')


class RobotArm(gym.Env):
    metadata = {'render.modes': ['human']}
    num_cubes = 1

    def __init__(self):
        self.pr = PyRep()
        self.reset()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.cubes) * 3 + len(self.arm.joints) + 1,),
            dtype=np.float32)

        self.action_space = gym.spaces.Discrete(len(self.arm.joints) * 2 + 2)

    def reset(self):
        if self.pr.running:
            self.pr.stop()
            self.pr.shutdown()
        self.pr.launch(SCENE_FILE, headless=True)
        self.pr.current_timestep = 0.0

        self.arm = Dobot()
        self.suction_cup_state = False

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

    def step(self, action):
        # Individual joint movements
        if action == 0:
            self.arm.joints[0].set_joint_target_velocity(-1.0)
        elif action == 1:
            self.arm.joints[0].set_joint_target_velocity(1.0)
        elif action == 2:
            self.arm.joints[1].set_joint_target_velocity(-1.0)
        elif action == 3:
            self.arm.joints[1].set_joint_target_velocity(1.0)
        elif action == 4:
            self.arm.joints[2].set_joint_target_velocity(-1.0)
        elif action == 5:
            self.arm.joints[2].set_joint_target_velocity(1.0)
        elif action == 6:
            self.arm.joints[3].set_joint_target_velocity(-1.0)
        elif action == 7:
            self.arm.joints[3].set_joint_target_velocity(1.0)
        # Suction cup
        elif action == 8:
            # Try to grasp each cube in the scene.
            if len(self.arm.suction_cup.get_grasped_objects()) == 0:
                for cube in self.cubes:
                    if self.arm.suction_cup.grasp(cube):
                        break
        else:
            self.arm.suction_cup.release()

        self.pr.step()
        self.pr.current_timestep += self.pr.get_simulation_timestep()

        reward = self._get_reward()
        done = self.pr.current_timestep > 16

        return self._create_observation(), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.pr.shutdown()

    def _create_observation(self):
        return np.array([
            *self._get_joint_positions(),
            1.0 if self.suction_cup_state else 0.0,
            *self._get_cube_positions(),
        ])

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
            bonus = -distance
            proximity_bonus += bonus

        return grasped_objects + proximity_bonus
