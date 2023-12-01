import collections
import gym
import numpy as np

from robohive.envs import env_base
from robohive.robot.robot_interbotix_arm import InterbotixArmRobot

class InterbotixSimpleEnv(env_base.MujocoEnv):
    
    def __init__(self, model_path, n_arms, arm_dof, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self.n_arms = n_arms
        self.arm_dof = arm_dof
        self._setup(robot_dof=arm_dof, robot_n_arms=n_arms, **kwargs)

    def _setup(self,
               robot_dof,
               robot_n_arms,
               frame_skip=40,
               reward_mode="dense",
               **kwargs,
        ):
        obs_keys = self._generate_obs_keys()
        proprio_keys = self._generate_proprio_keys()
        
        super()._setup(obs_keys=obs_keys,
                       proprio_keys=proprio_keys,
                       weighted_reward_keys={},
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       robot_class=InterbotixArmRobot,
                       robot_dof=robot_dof,
                       robot_n_arms=robot_n_arms,
                       **kwargs)

    def _generate_obs_keys(self):
        return self._generate_proprio_keys()
        
    def _generate_proprio_keys(self):
        prioprio_keys = []
        for i in range(self.n_arms):
            prioprio_keys.append(f'qp_arm{i}')
            prioprio_keys.append(f'qv_arm{i}')
        return prioprio_keys
        
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        
        for i in range(self.n_arms):
            arm_range = slice(i*self._arm_length, (i+1)*self._arm_length)
            arm_qpos = sim.data.qpos[arm_range].copy()
            arm_qvel = sim.data.qvel[arm_range].copy()
            obs_dict[f'qp_arm{i}'] = arm_qpos
            obs_dict[f'qv_arm{i}'] = arm_qvel
            
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
        obs = super().reset()
        return obs

    @property
    def _arm_length(self):
        return self.arm_dof + 3
