## Install
`pip install metaworld`

1. Main code is copied from the libero, consider a base class?
2. ReconfigureSubprocEnv can support autoreset mode but gym.vector.AsyncVectorEnv can't support, and the speed of two environments are comparable under the egl mode, but under the osmesa mode, the ReconfigureSubprocEnv environment is much slower.
3. egl render method is ugly by `os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.seed_offset)`
