import os

from gym.envs.registration import register

from robohive.envs.env_variants import register_env_variant
from robohive.envs.ur5.ur5_cube import Ur5CubeEnv


curr_dir = os.path.dirname(os.path.abspath(__file__))
print("RoboHive:> Registering UR5 Envs")


register(
    id='Ur5Cube-v0',
    entry_point='robohive.envs.ur5.ur5_cube:Ur5CubeEnv',
    kwargs={
        'model_path': curr_dir+'/assets/ur5_cube.xml',
        'config_path': curr_dir+'/assets/ur5_cube.config',
        'normalize_act': False,
        'visual_keys':[
            "rgb:top_down:160x240:2d",
            "rgb:front:160x240:2d",
            "rgb:gripper:160x240:2d",
        ],
    }
)
