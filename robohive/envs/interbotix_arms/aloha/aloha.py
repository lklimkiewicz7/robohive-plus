import collections
import gym
import numpy as np

from robohive.envs import env_base

class AlohaEnv(env_base.MujocoEnv):
    
    CAMERA_SIZE = '80x128'
    CAMERA_ENCODER = '2d'
    ARM_NDOF = 8
    
    DEFAULT_OBS_KEYS = [
        'qp_arm0', 'qp_arm1', 'qv_arm0', 'qv_arm1'
    ]
    DEFAULT_PROPRIO_KEYS = [
        'qp_arm0', 'qp_arm1', 'qv_arm0', 'qv_arm1'
    ]
    DEFAULT_VISUAL_KEYS = [
        f'rgb:top_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
        f'rgb:front_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
        f'rgb:vx300_arm0_gripper_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
        f'rgb:vx300_arm1_gripper_cam:{CAMERA_SIZE}:{CAMERA_ENCODER}',
    ]

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)


    def _setup(self,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               proprio_keys=DEFAULT_PROPRIO_KEYS,
               **kwargs,
        ):
        super()._setup(obs_keys=obs_keys,
                       proprio_keys=proprio_keys,
                       weighted_reward_keys={},
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp_arm0'] = sim.data.qpos[:self.ARM_NDOF].copy()
        obs_dict['qv_arm0'] = sim.data.qvel[:self.ARM_NDOF].copy()
        obs_dict['qp_arm1'] = sim.data.qpos[self.ARM_NDOF:2*self.ARM_NDOF].copy()
        obs_dict['qv_arm1'] = sim.data.qvel[self.ARM_NDOF:2*self.ARM_NDOF].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict((
            ('sparse',  0),
            ('solved',  False),
            ('done',    False),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
