from robohive.robot.hardware_base import hardwareBase

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


class InterbotixArm(hardwareBase):
    def __init__(self, name, arm_model, ip_address, **kwargs):
        self.name = name
        self.arm_model = arm_model
        self.ip_address = ip_address
        
        self.bot = None

    def connect(self):
        self.bot = InterbotixManipulatorXS(
            robot_model=self.arm_model,
            robot_name=self.name
        )

    def okay(self):
        okay = False
        try:
            self.bot.core.robot_get_joint_states()
        except Exception:
            okay = False
            self.bot = None
        return okay

    def close(self):
        if self.bot is not None:
            self.bot.shutdown()

    def get_sensors(self):
        sensors = np.array(self.bot.core.robot_get_joint_states().position)
        return sensors

    def reset(self, reset_pos=None):
        self.bot.arm.set_joint_positions(reset_pos[:5], blocking=True)

    def apply_commands(self, q_desired):
        self.bot.arm.set_joint_positions(q_desired[:5], blocking=False)

    def __del__(self):
        self.close()
