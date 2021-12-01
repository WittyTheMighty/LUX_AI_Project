import numpy as np
import gym
from stable_baselines3 import PPO
import Lux_Project_Env.frozen_lake as fl

if __name__ == '__main__':
    env = fl.FrozenLakeEnv(is_slippery=False)
    model = PPO('MlpPolicy', env, verbose=1)
    # model = PPO.load('./model.zip')
    # model.set_env(env)
    model.learn(total_timesteps=10000)
    model.save('./model.zip')

    obs = env.reset()
    env.render()
    for i in range(100):
        action, _state = model.predict(obs)
        obs, r, done, info = env.step(action)
        env.render()
        if done : break