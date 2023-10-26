import os

from gym.envs.registration import register
from robohive.envs.arms.push_base_v0 import PushBaseV0


curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Interbotix Arms Envs")

register(
    id='InterbotixVX300PushFixed-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/vx300/vx300_ycb_v0.xml',
        'config_path': curr_dir+'/vx300/vx300_ycb_v0.config',
        'robot_ndof': 9,
        'robot_site_name': "end_effector",
        'object_site_name': "sugarbox",
        'target_site_name': "target",
        'target_xyz_range': {'high':[-.4, 0.5, 0.78], 'low':[-.4, 0.5, 0.78]}
    }
)