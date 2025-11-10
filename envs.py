import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from collections import deque
import random

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
    
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(k, 84, 84), dtype=np.uint8)

    def reset(self):
        ob,_ = self.env.reset()
        for _ in range(self.k):
            self.frames.append(self.process_frame(ob))
        return self._get_ob()

    def step(self, action):
        ob, reward, done, _ , info = self.env.step(action)
        if isinstance(ob, tuple):
            ob = ob[0]
        self.frames.append(self.process_frame(ob))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
    
    def process_frame(self,frame):
        """Preprocess a 210x160x3 frame to 84x84x1 grayscale image."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.uint8)
        frame = np.expand_dims(frame, axis=0)
        return frame  # Add channel dimension
    
def wrap_deepmind(env_id,frame_stack=4,render_mode=None):
    """Configure environment for DeepMind-style Atari."""
    env = gym.make(env_id,render_mode=render_mode)
    env = FrameStack(env, frame_stack)
    return env

