import collections
import numpy as np

from robohive.envs.interbotix_arms.itx_simple import InterbotixSimpleEnv
from robohive.robot.robot_interbotix_arm import MODELS_DOF, MODELS_SLEEP_POSES

class InterbotixCubesEnv(InterbotixSimpleEnv):
    
    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict['red_qpos'] = np.array([1, 2, 3])
        obs_dict['green_qpos'] = np.array([1, 2, 3])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        rwd_dict = collections.OrderedDict((
            ('sparse',  0),
            ('solved',  False),
            ('done',    False),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def _generate_proprio_keys(self):
        return super()._generate_proprio_keys() + 