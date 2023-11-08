import math

from .robot import Robot


MODELS_DOF = {
    'vx300': 5,
    'vx300s': 6
}

MODELS_SLEEP_POSES = {
    'vx300': [0.0, -1.85, 1.54, 0.8, 0.0, 0.0, 0.0, 0.0],
    'vx300s': [0.0, -1.85, 1.55, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]
}


class InterbotixArmRobot(Robot):
    
    DEFAULT_HORN_RADIUS = 0.022
    DEFAULT_ARM_LENGTH = 0.036
    
    def __init__(
            self,
            robot_dof,
            *args,
            robot_n_arms = 1,
            robot_horn_radius = DEFAULT_HORN_RADIUS,
            robot_arm_length = DEFAULT_ARM_LENGTH,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.dof = robot_dof
        self.n_arms = robot_n_arms
        self.horn_radius = robot_horn_radius
        self.arm_length = robot_arm_length
    
    def transform_ctrl(self, ctrl):
        per_arm = self.dof + 3
        assert len(ctrl) == per_arm * self.n_arms
        
        new_ctrl = []
        
        for i in range(self.n_arms):
            arm_ctrl = ctrl[i*per_arm : (i+1)*per_arm]
            gripper_angle = arm_ctrl[-3]
            gripper_linear = self._angular_to_linear(gripper_angle)
            arm_ctrl[-2:] = [gripper_linear, -gripper_linear]
            new_ctrl.extend(arm_ctrl)
        
        assert len(new_ctrl) == per_arm * self.n_arms
        return new_ctrl
    
    def _angular_to_linear(self, angular_position):
        a1 = self.horn_radius * math.sin(angular_position)
        c = math.sqrt(self.horn_radius**2 - a1**2)
        a2 = math.sqrt(self.arm_length**2 - c**2)
        return a1 + a2
