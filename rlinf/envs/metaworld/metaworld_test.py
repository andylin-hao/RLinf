import gymnasium as gym
import metaworld

seed = 42

envs = gym.make_vec('Meta-World/MT50', vector_strategy='async', seed=seed) # this returns a Synchronous Vector Environment with 50 environments

obs, info = envs.reset() # reset all 50 environments

a = envs.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 50 environments

print(obs.shape)
