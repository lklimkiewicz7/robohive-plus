import os

from gym.envs.registration import register

from robohive.envs.env_variants import register_env_variant
from robohive.envs.arms.reach_base_v0 import ReachBaseV0


curr_dir = os.path.dirname(os.path.abspath(__file__))
print("RoboHive:> Registering UR5 Envs")


register(
    id='Ur5ReachFixed-v0',
    entry_point='robohive.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=1000,
    kwargs={
        'model_path': curr_dir+'/assets/ur5_reach.xml',
        'config_path': curr_dir+'/assets/ur5_reach.config',
        'robot_site_name': "end_effector",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.2, 0.3, 1.2], 'low':[0.2, 0.3, 1.2]},
        'normalize_act': False,
    }
)

register_env_variant(
    env_id='Ur5ReachFixed-v0',
    variant_id='Ur5ReachFixedCam-v0',
    variants={
            'visual_keys':[
                "rgb:top_down:224x224:2d",
                "rgb:front:224x224:2d",
            ],
        },
    silent=True
)
