from pyrep.objects import Joint
from pyrep.robots.end_effectors.dobot_suction_cup import DobotSuctionCup


class Dobot:
    def __init__(self):
        self.joints = [Joint('Dobot_joint{}'.format(i + 1)) for i in range(4)]
        for joint in self.joints:
            joint.set_control_loop_enabled(False)

        self.suction_cup = DobotSuctionCup()
