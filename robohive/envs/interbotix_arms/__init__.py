import os

from gym.envs.registration import register
from robohive.envs.env_variants import register_env_variant
from .itx_simple import InterbotixSimpleEnv
from .itx_cubes import InterbotixCubesEnv


curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Interbotix Arms Envs")


CAMERA_SIZE = '80x128'
CAMERA_ENCODER = '2d'


register(
    id='InterbotixVx300sDualSimple-v0',
    entry_point='robohive.envs.interbotix_arms.itx_simple:InterbotixSimpleEnv',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/assets/itx_vx300s_dual.xml',
        'config_path': curr_dir+'/assets/itx_vx300s_dual.config',
        'n_arms': 2,
        'arm_model': 'vx300s'
    },
)


register(
    id='InterbotixVx300DualSimple-v0',
    entry_point='robohive.envs.interbotix_arms.itx_simple:InterbotixSimpleEnv',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/assets/itx_vx300_dual.xml',
        'config_path': curr_dir+'/assets/itx_vx300_dual.config',
        'n_arms': 2,
        'arm_model': 'vx300'
    },
)


register(
    id='InterbotixVx300sDualCamSimple-v0',
    entry_point='robohive.envs.interbotix_arms.itx_simple:InterbotixSimpleEnv',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/assets/itx_vx300s_dual.xml',
        'config_path': curr_dir+'/assets/itx_vx300s_dual.config',
        'n_arms': 2,
        'arm_model': 'vx300s',
        'visual_keys': [
            f'rgb:top_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
            f'rgb:front_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
            f'rgb:vx300s_arm0_gripper_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
            f'rgb:vx300s_arm1_gripper_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
        ]
    },
)


register(
    id='InterbotixVx300DualCamSimple-v0',
    entry_point='robohive.envs.interbotix_arms.itx_simple:InterbotixSimpleEnv',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/assets/itx_vx300_dual.xml',
        'config_path': curr_dir+'/assets/itx_vx300_dual.config',
        'n_arms': 2,
        'arm_model': 'vx300',
        'visual_keys': [
            f'rgb:top_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
            f'rgb:front_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
            f'rgb:vx300_arm0_gripper_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
            f'rgb:vx300_arm1_gripper_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
        ]
    },
)


register(
    id='InterbotixVx300sSimple-v0',
    entry_point='robohive.envs.interbotix_arms.itx_simple:InterbotixSimpleEnv',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/assets/itx_vx300s.xml',
        'config_path': curr_dir+'/assets/itx_vx300s.config',
        'n_arms': 1,
        'arm_model': 'vx300s'
    },
)


register(
    id='InterbotixVx300sDualBlocksCam-v0',
    entry_point='robohive.envs.interbotix_arms.itx_cubes:InterbotixCubesEnv',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/assets/itx_vx300s_dual_blocks.xml',
        'config_path': curr_dir+'/assets/itx_vx300s_dual_blocks.config',
        'n_arms': 2,
        'arm_model': 'vx300s'
    },
)