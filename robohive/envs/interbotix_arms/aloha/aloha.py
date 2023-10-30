import collections
import gym
import numpy as np

from robohive.envs import env_base

class AlohaEnv(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp_robot', 'qv_robot'
    ]
    DEFAULT_PROPRIO_KEYS = [
        'qp_robot', 'qv_robot',
    ]
    DEFAULT_VISUAL_KEYS = [
        'rgb:top_cam:160x256:2d',
        'rgb:front_cam:160x256:2d',
        'rgb:vx300_arm0_gripper_cam:160x256:2d',
        'rgb:vx300_arm1_gripper_cam:160x256:2d',
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {}

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)


    def _setup(self,
               robot_ndof,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               proprio_keys=DEFAULT_PROPRIO_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):
        self.robot_ndof = robot_ndof
        super()._setup(obs_keys=obs_keys,
                       proprio_keys=proprio_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp_robot'] = sim.data.qpos[:self.robot_ndof].copy()
        obs_dict['qv_robot'] = sim.data.qvel[:self.robot_ndof].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict((
            # Must keys
            ('sparse',  0),
            ('solved',  False),
            ('done',    False),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
