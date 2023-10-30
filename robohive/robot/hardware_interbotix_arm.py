from robohive.robot.hardware_base import hardwareBase


class InterbotixArm(hardwareBase):
    def __init__(self, name, arm_model, ip_address, **kwargs):
        self.name = name
        self.arm_model = arm_model
        self.ip_address = ip_address

    def connect(self):
        print('Connecting')

    def okay(self):
        print('Okey')
        return True

    def close(self):
        print('Closing')

    def reset(self, reset_pos=None, time_to_go=5):
        print('Reseting')

    def get_sensors(self):
        print('Getting sensors')
        return [0]*8

    def apply_commands(self, q_desired):
        print('Applying commands')

    def __del__(self):
        self.close()
