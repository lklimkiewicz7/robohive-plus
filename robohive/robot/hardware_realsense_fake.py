import numpy as np
# from hardware_base import hardwareBase
from robohive.robot.hardware_base import hardwareBase

import a0
import logging
import copy
import time
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import robohive.robot.serdes as serdes

sensor_msgs = serdes.get_capnp_msgs('sensor_msgs')

class FakeRealSense(hardwareBase):
    def __init__(self, name, rgb_topic=None, d_topic=None, **kwargs):
        assert rgb_topic or d_topic, "Atleast one of the topics is needed"
        self.rgb_topic = rgb_topic
        self.d_topic = d_topic
        self.last_image_pkt = None
        self.rgb_sub = None
        self.last_depth_pkt = None
        self.most_recent_pkt_ts = datetime.datetime.now()
        self.timeout = 1 # in seconds

    def connect(self):
        print("Fake realsense connected")

    def get_sensors(self):
        # get all data from all topics
        image = np.zeros((80, 128, 3), dtype=np.float32)
        return {'time': datetime.datetime.now(), 'rgb': image, 'd': image}

    def apply_commands(self):
        return 0

    def close(self):
        print("Fake realsense disconnected")
        return True

    def okay(self):
        print("Fake realsense okay")
        return True

    def reset(self):
        return 0
