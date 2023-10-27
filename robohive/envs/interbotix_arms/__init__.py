import os

from gym.envs.registration import register
from robohive.envs.env_variants import register_env_variant
from .aloha.aloha import AlohaEnv


curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Interbotix Arms Envs")

register(
    id='Aloha-v0',
    entry_point='robohive.envs.interbotix_arms.aloha.aloha:AlohaEnv',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/aloha/aloha.xml',
        'config_path': curr_dir+'/aloha/aloha.config',
        'robot_ndof': 9,
        'robot_site_name': "vx300_arm0_end_effector",
        'object_site_name': "sugarbox",
        'target_site_name': "target",
        'target_xyz_range': {'high':[-.4, 0.5, 0.78], 'low':[-.4, 0.5, 0.78]},
        'visual_keys': AlohaEnv.DEFAULT_VISUAL_KEYS,
    },
)
