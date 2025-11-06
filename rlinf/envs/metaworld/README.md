# Metaworld Environment Setup Guide

## To RLinf authors:
1. The main code of `metaworld_env.py` was adapted from the Libero environment. Would you consider implementing a base environment class so that other environments can inherit from it?
2. I modified a version of `ReconfigureSubprocEnv` (originally from Libero) to support autoreset mode. However, I noticed that this environment runs significantly slower than `gym.vector.AsyncVectorEnv` under the osmesa mode. In the `egl` mode, their performance is comparable.
3. Currently, the EGL rendering method is inelegantly set using `os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.seed_offset)`. Is there a better approach to manage this?

## Installation

Install the Metaworld environment with:
```
pip install metaworld
```

## Running Environment Tests
Run the test script with:
```
python metaworld_test.py
```

## Getting Help
If you encounter issues not addressed in this guide, please:
1. Refer to the [Metaworld documentation](https://metaworld.farama.org/)
2. Create an issue in the RLinf repository

## License
Please refer to the individual repository licenses for the Metaworld.
