import os

from gym.envs.registration import register

from robohive.envs.env_variants import register_env_variant
from robohive.envs.ur5.ur5_env import Ur5Env


curr_dir = os.path.dirname(os.path.abspath(__file__))
print("RoboHive:> Registering UR5 Envs")


register(
    id='Ur5ReachFixed-v0',
    entry_point='robohive.envs.ur5.ur5_env:Ur5Env',
    kwargs={
        'model_path': curr_dir+'/assets/ur5_reach.xml',
        'config_path': curr_dir+'/assets/ur5_reach.config',
        'normalize_act': False,
    }
)

register_env_variant(
    env_id='Ur5ReachFixed-v0',
    variant_id='Ur5ReachFixedCam-v0',
    variants={
            'visual_keys':[
                "rgb:top_down:160x240:2d",
                "rgb:front:160x240:2d",
                "rgb:gripper:160x240:2d",
            ],
        },
    silent=True
)
