import collections
import random
from math import pi

import gym
import numpy as np

from robohive.envs import env_base

class Ur5CubeEnv(env_base.MujocoEnv):
    
    INITIAL_POS = [-2.30, -2.34, 1.85, -1.06, -1.57, -0.73, 0.054, 0.054]
    DEFAULT_OBS_KEYS = ['position', 'velocity']
    DEFAULT_PROPRIO_KEYS = ['position']
    
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
        
        self.init_qpos = self.INITIAL_POS
        self.reset()
        
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['position'] = sim.data.qpos.copy()[:7]
        obs_dict['velocity'] = sim.data.qvel.copy()[:7]
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
        object_pos = self._random_position()
        object_rot = self._random_rotation()
        object_data = object_pos + [1] + object_rot
        obs = super().reset(reset_qpos=self.INITIAL_POS + object_data)
        
        self.sim.data.ctrl[:] = self.INITIAL_POS[:-1]
        self.sim.advance(substeps=500, render=False)
        
        return obs
    
    def _random_position(self):
        return [
            random.uniform(0.3, 0.7),
            random.uniform(-0.45, 0.45),
            0.1
        ]
        
    def _random_rotation(self):
        return [
            random.uniform(-pi, pi),
            random.uniform(-pi, pi),
            random.uniform(-pi, pi),
        ]
