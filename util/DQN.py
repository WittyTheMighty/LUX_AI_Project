import copy
from collections import deque
from time import sleep

import gym
import numpy as np
import torch
import random

from matplotlib import pyplot as plt
from torch import nn

from Lux_Project_Env import frozen_lake


class DQN_Agent:

    def __init__(self, config):
        torch.manual_seed(1423)
        self.batch_size = config['batch_size']
        self.state_dim = config['state_dim']
        self.subgoal_dim = config['subgoal_dim']
        self.epsilon = config['epsilon']
        self.epsilon_disc = config['epsilon_disc']
        self.K = config['K']
        # self.q_net = self.build_nn(layer_sizes)
        self.q_net = self.build_nn_2()
        self.target_net = copy.deepcopy(self.q_net)
        # self.q_net
        # self.target_net
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config['lr'])

        self.network_sync_freq = config['sync_freq']
        self.network_sync_counter = 0
        self.gamma = torch.tensor(config['gamma']).float()
        self.experience_replay = deque(maxlen=config['exp_replay_size'])
        return

    def discount_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon -= self.epsilon_disc

    def build_nn_2(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            # nn.LSTM(4, 64),
            nn.Linear(64, self.subgoal_dim),
            nn.Identity(),
        )
        return model

    def policy(self, state):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            if self.state_dim == 1:
                state = np.array([state])
            Qp = self.q_net(torch.from_numpy(state).float())
        Q, A = torch.max(Qp, dim=0)
        A = A if torch.rand(1, ).item() > self.epsilon else torch.randint(0, self.subgoal_dim, (1,))
        return A.item()

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, dim=1)
        return q

    def record(self, obs, action, reward, obs_next):
        self.experience_replay.append((obs, action, reward, obs_next))
        return

    def sample_from_experience(self, sample_size):
        if (len(self.experience_replay) < sample_size):
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def learn(self):

        for _ in range(self.K):
            s, a, rn, sn = self.sample_from_experience(sample_size=self.batch_size)
            if (self.network_sync_counter == self.network_sync_freq):
                self.target_net.load_state_dict(self.q_net.state_dict())
                self.network_sync_counter = 0

            # predict expected return of current state using main network
            if self.state_dim == 1:
                s = s.reshape((-1,1))
                sn = sn.reshape((-1,1))
            qp = self.q_net(s)
            pred_return, _ = torch.max(qp, dim=1)

            # get target return using target network
            q_next = self.get_q_next(sn)
            target_return = rn + self.gamma * q_next

            loss = self.loss_fn(pred_return, target_return)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            self.network_sync_counter += 1
        agent.discount_epsilon()
        return

if __name__=='__main__':
    # env = gym.make('CartPole-v1')
    env = frozen_lake.FrozenLakeEnv(is_slippery=False)
    exp_replay_size = 256
    config = {
        'state_dim': 1,
        'subgoal_dim': 4,
        'lr': 5e-3,         # higher lr seems better
        'sync_freq': 5,
        'exp_replay_size': exp_replay_size,
        'batch_size': 16,
        'epsilon': 1,
        'epsilon_disc': (1/5000), # around n_episodes / 2 or /3
        'K': 1,
        'gamma': 0.95
    }
    agent = DQN_Agent(config)

    # Main training loop
    losses_list, reward_list, episode_len_list,= [], [], []
    index = 128
    episodes = 15000
    avg_reward_list = []
    for i in range(episodes):
        obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
        while (done != True):
            ep_len += 1
            A = agent.policy(obs)
            obs_next, reward, done, _ = env.step(A)
            agent.record(obs, A, reward, obs_next)

            obs = obs_next
            rew += reward
            index += 1

        agent.learn()
        reward_list.append(rew), episode_len_list.append(ep_len)
        mean = np.round(np.mean(reward_list[:-40]), 2)
        avg_reward_list.append(mean)
        print(mean)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()