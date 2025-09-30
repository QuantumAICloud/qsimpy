import gymnasium as gym
from gymnasium.core import Env
# Make sure these are here if not already
from numpy.random import default_rng
import simpy
import numpy as np


class ScaleQSimPyEnv(gym.RewardWrapper):
    def __init__(self, env: Env, scale: float):
        super().__init__(env)
        self.scaling_factor = scale

    def reward(self, reward):
        reward *= self.scaling_factor
        return reward
class SerializableEnvWrapper(gym.Wrapper):
    def __getattr__(self, name):
        return getattr(self.env, name)

    def __getstate__(self):
        # Start with wrapper __dict__
        state = self.__dict__.copy()

        # Replace self.env with its safe state
        if hasattr(self.env, "__getstate__"):
            state["env_state"] = self.env.__getstate__()
        else:
            state["env_state"] = self.env.__dict__.copy()

        # Don't pickle the actual env object directly
        if "env" in state:
            del state["env"]

        # Debug: check for generators in wrapper state
        for k, v in list(state.items()):
            if hasattr(v, "__iter__") and not isinstance(v, (list, tuple, dict, str, bytes, np.ndarray)):
                print(f"[WRAPPER-PICKLE] Removing generator-like object at key '{k}' ({type(v)})")
                del state[k]

        # Preserve dataset path for reconstruction
        state["_dataset_path"] = getattr(self.env, "dataset_path", None)
        return state

    def __setstate__(self, state):
        from gymenv_qsimpy import QSimPyEnv
        dataset_path = state.pop("_dataset_path", None)
        if not dataset_path:
            raise ValueError("Missing dataset path for deserialization")

        new_env = QSimPyEnv(dataset=dataset_path)
        if "env_state" in state:
            new_env.__setstate__(state.pop("env_state"))

        super().__init__(new_env)
        self.__dict__.update(state)
