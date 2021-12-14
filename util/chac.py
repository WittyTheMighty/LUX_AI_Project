import os

import numpy as np

from util.ppo_v2 import PPO
import torch
# from DDPG import DDPG
from util.DQN import DQN_Agent


class CompositeActorCritic:
    """"
    " config: dict with DDPG and PPO parameters
    """

    def __init__(self, env, config, mode="train", use_cuda=False):
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.env = env
        self.low_level_agent = PPO(config["PPO"])
        # self.high_level_agent = DDPG(config["DDPG"])
        self.high_level_agent = DQN_Agent(config["DQN"])

        self.batch_size = config['batch_size']

        self.n_episodes = config["n_episodes"]
        # Hyperparameter before updating PPO policy
        self.attempts = config["attempts"]

        self.step_limit = config["step_limit"]

        self.internal_reward = 0
        # id : subgoal
        self.subgoal_unit_tracker = {}

    def train_CHAC(self):

        for episode in range(self.n_episodes):
            reward = 0
            # Env reset return the initial environment
            next_state = s_l = self.env.reset()
            # [
            sub_goal = self.high_level_agent.policy(s_l)  # Noise already included in DDPG policy
            current_unit_id = self.env.last_observation_object[0].id
            self.subgoal_unit_tracker[current_unit_id] = sub_goal

            # Sub-set of episodes
            for T in range(self.attempts):
                # Concatenation sub-goal vectors need to represent thing we can control like:
                # -Ressources
                # -Build house
                # -Collect research point
                # The neural network will take the concatenation between both vectors
                state = next_state
                action = self.low_level_agent.policy(state, sub_goal)  # TODO : concatenation
                next_state, env_reward, done, _ = self.env.step(action)
                if self.env.last_observation_object[0] is None:
                    current_unit_id = self.env.last_observation_object[1].get_tile_id()
                else:
                    current_unit_id = self.env.last_observation_object[0].id

                if done:
                    next_state = state

                sub_goal = self.subgoal_unit_tracker.get(current_unit_id)
                if sub_goal is None:
                    sub_goal = self.high_level_agent.policy(next_state)
                    self.subgoal_unit_tracker[current_unit_id] = sub_goal

                # Give reward if agent reaches state
                internal_reward = self.compute_internal_reward(sub_goal, next_state)

                # Reward to setup in luxAi Agent policy
                reward = reward + env_reward

                self.low_level_agent.record(internal_reward, done)  # agents already stores states and actions
                if self.subgoal_reached(sub_goal, next_state) or T > self.step_limit:
                    if self.subgoal_reached(sub_goal, next_state):
                        reward += internal_reward

                    # Not clear what state to store here sL
                    self.high_level_agent.record(next_state, sub_goal, reward, state)
                    s_l = next_state
                    sub_goal = self.high_level_agent.policy(s_l)  # noise high policy algo
                    reward = 0

                if done:
                    break

            self.low_level_agent.learn()
            self.low_level_agent.buffer.clear()

            if len(self.high_level_agent.experience_replay) > self.batch_size:
                self.high_level_agent.learn()

    def predict(self, observation, unit_id):
        sub_goal = self.high_level_agent.policy(observation)
        self.subgoal_unit_tracker[unit_id] = sub_goal
        return self.low_level_agent.policy(observation, sub_goal)

    def observation_to_subgoal(self, observation):
        subgoal_observation = np.zeros(5)
        # Number of citytiles
        subgoal_observation[0] = observation[-11]
        # Number of workers / Number of carts
        subgoal_observation[1] = observation[-7]
        subgoal_observation[2] = observation[-9]
        # Number of research points
        subgoal_observation[3] = observation[-5]
        # Fuel on the current unit
        subgoal_observation[4] = observation[-1]

        return subgoal_observation

    def subgoal_reached(self, sub_goal, observation):
        sub_goal_observation = self.observation_to_subgoal(observation)

        for goal, obs in zip(sub_goal, sub_goal_observation):
            if obs <= goal:
                return False
        return True

    # The internal reward system may be subject to change in
    # the luxAI environment
    def compute_internal_reward(self, sub_goal, observation):
        if self.subgoal_reached(sub_goal, observation):
            return 1
        else:
            return 0

    def save_checkpoint(self, path):
        torch.save(self.high_level_agent.q_net, path + "-DDPG-actor.pt")
        torch.save(self.high_level_agent.target_net, path + "-DDPG-critic.pt")
        torch.save(self.low_level_agent.policy_, path + "-PPO.pt")

    def load_checkpoint(self, path):
        self.high_level_agent.q_net = torch.load(path + "-DDPG-actor.pt")
        self.high_level_agent.target_net = torch.load(path + "-DDPG-critic.pt")
        self.low_level_agent.policy_ = torch.load(path + "-PPO.pt")
