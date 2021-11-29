import numpy as np
import gym
from stable_baselines3 import PPO
import frozen_lake as fl

if __name__ == '__main__':
    env = fl.FrozenLakeEnv(map_name='4x4')
    model = PPO('MlpPolicy', env, verbose=1)
    # model = PPO.load('./model.zip')
    # model.set_env(env)
    model.learn(total_timesteps=50000)
    model.save('./model.zip')

    obs = env.reset()
    # env.render()
    rewards = 0
    for i in range(10000):
        action, _state = model.predict(obs)
        obs, r, done, info = env.step(action)
        rewards += r
        # env.render()
        if done : break
    print(rewards)