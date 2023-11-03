import math

import numpy as np

from .robot import Robot

horn_radius = 0.022
arm_length = 0.036


class InterbotixArmRobot(Robot):
    
    HORN_RADIUS = 0.022
    ARM_LENGTH = 0.036
    
    def transform_ctrl(self, pos):
        gripper_1 = self._angular_to_linear(pos[5])
        gripper_2 = self._angular_to_linear(pos[13])
        
        pos[6:8] = [-gripper_1, gripper_1]
        pos[14:16] = [-gripper_2, gripper_2]
        return pos
    
    def _angular_to_linear(self, angular_position):
        a1 = self.HORN_RADIUS * math.sin(angular_position)
        c = math.sqrt(self.HORN_RADIUS**2 - a1**2)
        a2 = math.sqrt(self.ARM_LENGTH**2 - c**2)
        return a1 + a2
