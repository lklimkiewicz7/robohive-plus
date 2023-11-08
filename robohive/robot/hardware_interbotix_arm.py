import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

from robohive.robot.hardware_base import hardwareBase
from robohive.robot.robot_interbotix_arm import MODELS_DOF


class InterbotixArm(hardwareBase):
    def __init__(self, name, arm_model, ip_address, **kwargs):
        self.name = name
        self.arm_model = arm_model
        self.ip_address = ip_address
        self.dof = MODELS_DOF[arm_model]
        
        self.bot = None

    def connect(self):
        self.bot = InterbotixManipulatorXS(
            robot_model=self.arm_model,
            robot_name=self.name,
            gripper_name=None
        )
        self.bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")

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
            self.bot.core.robot_set_operating_modes("single", "gripper", "pwm")
            self.bot.shutdown()

    def get_sensors(self):
        sensors = np.array(self.bot.core.robot_get_joint_states().position)
        assert len(sensors) == self.dof + 3
        return sensors

    def reset(self, reset_pos):
        assert len(reset_pos) == self.dof + 3
        self.bot.core.robot_write_joint_command("gripper", reset_pos[self.dof])
        self.bot.arm.set_joint_positions(reset_pos[:self.dof], blocking=True)

    def apply_commands(self, q_desired):
        assert len(q_desired) == self.dof + 3
        self.bot.arm.set_joint_positions(q_desired[:self.dof], blocking=False)
        self.bot.core.robot_write_joint_command("gripper", q_desired[self.dof])

    def __del__(self):
        self.close()
