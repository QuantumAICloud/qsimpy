from gymenv_qsimpy import QSimPyEnv
from env_wrapper import ScaleQSimPyEnv , SerializableEnvWrapper
from gymnasium.wrappers import RescaleObservation, DtypeObservation
import numpy as np


def qsimpy_env_creator(env_config):
    dataset = env_config.pop("dataset", None)
    config = env_config.pop("config", None)
    config = config if config is not None else {}
    if dataset is None:
        raise ValueError("Dataset is not specified")
    
    env = QSimPyEnv(dataset=dataset, config=config)
    env = SerializableEnvWrapper(env)
    
    obs_filter = env_config.pop("obs_filter", None)
    reward_filter = env_config.pop("reward_filter", None)

    if obs_filter is not None:
        if obs_filter == "rescale_-1_1":
            env = RescaleObservation(
                env=env,
                min_obs=np.ones((env.obs_dim,), dtype=np.float32) * -1,
                max_obs=np.ones((env.obs_dim,), dtype=np.float32) * 1,
            )
            env = DtypeObservation(env, dtype=np.float32)

    if reward_filter is not None:
        if reward_filter == "scale_2x":
            env = ScaleQSimPyEnv(env, scale=env_config.pop("reward_scale", 2))
            
    # for i in range(10) : print(type(env))
    return env