import gym
from gym import spaces
import numpy as np


class Ur5CubeOctoJointsEnv(gym.Env):
    
    def __init__(self):
        self.env = gym.make('Ur5Cube-v0')
        
        self.JOINTS_LOW = np.array([-6.28319,  -6.28319,   -3.1415,    -6.28319,   -6.28319,   -6.28319,   0])
        self.JOINTS_HIGH = np.array([6.28319,  6.28319,    3.1415,     6.28319,    6.28319,    6.28319,    0.055])
        
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(low=self.JOINTS_LOW, high=self.JOINTS_HIGH, dtype=np.float32),
                "image_primary": spaces.Box(0, 255, shape=(160, 240, 3), dtype=np.uint8),
                "image_secondary": spaces.Box(0, 255, shape=(160, 240, 3), dtype=np.uint8),
                "image_wrist": spaces.Box(0, 255, shape=(160, 240, 3), dtype=np.uint8),
            }
        )
        
        self.action_space = spaces.Box(low=self.JOINTS_LOW, high=self.JOINTS_HIGH, dtype=np.float32)
    
    def _get_obs(self):
        visual_dict = self.env.get_exteroception()
        return {
            'proprio': np.array(self.env.obs_dict['position'], dtype=np.float32),
            'image_primary': visual_dict['rgb:front:160x240:2d'],
            'image_secondary': visual_dict['rgb:top_down:160x240:2d'],
            'image_wrist': visual_dict['rgb:gripper:160x240:2d']
        }
        
    def _get_info(self):
        return {}
        
    def reset(self, seed=None, options=None):
        self.env.reset()
        return self._get_obs(), self._get_info()
        
    def step(self, action):
        self.env.step(np.array(action))
        
        observation = self._get_obs()
        info = self._get_info()
        reward = 0
        done = False
        
        return observation, reward, done, info

    def get_task(self):
        return {
            'language_instruction': 'grasp cube'
        }



class Ur5CubeOctoGripperEnv(gym.Env):
    
    def __init__(self):
        self._env = gym.make('Ur5Cube-v0')
        
        self.PROPRIO_LOW = np.array([-1,  -1,   -0.2,    -1,   -1,   -1,   -1, 0])
        self.PROPRIO_HIGH = np.array([1,  1,    0.5,     1,    1,    1,    1,  1])
        
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(low=self.PROPRIO_LOW, high=self.PROPRIO_HIGH, dtype=np.float32),
                "image_primary": spaces.Box(0, 255, shape=(160, 240, 3), dtype=np.uint8),
                "image_secondary": spaces.Box(0, 255, shape=(160, 240, 3), dtype=np.uint8),
                "image_wrist": spaces.Box(0, 255, shape=(160, 240, 3), dtype=np.uint8),
            }
        )
        
        self.action_space = spaces.Box(low=self.PROPRIO_LOW, high=self.PROPRIO_HIGH, dtype=np.float32)
    
    def _get_obs(self):
        visual_dict = self._env.get_exteroception()
        gripper_state = 1 if self._env.obs_dict['position'][-1] < 0.045 else 0
        return {
            'proprio': np.concatenate([
                self._env.obs_dict['ee_position'], self._env.obs_dict['ee_orientation'], np.array([gripper_state]),  
            ], dtype=np.float32),
            'image_primary': visual_dict['rgb:front:160x240:2d'],
            'image_secondary': visual_dict['rgb:top_down:160x240:2d'],
            'image_wrist': visual_dict['rgb:gripper:160x240:2d']
        }
        
    def _get_info(self):
        return {}
        
    def reset(self, seed=None, options=None):
        self._env.reset()
        return self._get_obs(), self._get_info()
        
    def step(self, ee_action):
        position = ee_action[:3]
        orientation = ee_action[3:7]
        gripper = 0 if ee_action[7] > 0.5 else 0.055
        
        joints = self._env.robot.inverse_kinematics(position, orientation)
        if joints is None:
            print("Inverse kinematics failed. Unable to move to desired position. Doing nothing.")
        else:
            joint_action = np.concatenate([joints, np.array([gripper])])
            self._env.step(np.array(joint_action))
        
        observation = self._get_obs()
        info = self._get_info()
        reward = 0
        done = False
        
        return observation, reward, done, info

    def get_task(self):
        return {
            'language_instruction': 'grasp cube'
        }