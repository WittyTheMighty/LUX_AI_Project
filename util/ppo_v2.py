import os

import numpy as np
from matplotlib import pyplot as plt

import gym

# Adaptation from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from Lux_Project_Env import frozen_lake

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.subgoals = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.subgoals[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

#
# class PrintLayer(nn.Module):
#     def __init__(self):
#         super(PrintLayer, self).__init__()
#
#     def forward(self, x):
#         # Do your print / debug stuff here
#         print(x)
#         return x
#

class ActorCritic(nn.Module):
    def __init__(self, state_dim, subgoal_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim
        self.subgoal_dim = subgoal_dim
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim+subgoal_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim+subgoal_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim+subgoal_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state_subgoal):
        # s = state.clone().reshape(-1, 1)
        if self.has_continuous_action_space:
            action_mean = self.actor(state_subgoal)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # if self.state_dim == 1:
            #     state = state.reshape(-1,1)
            action_probs = self.actor(state_subgoal)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state_subgoal, action):
        # s = state.clone().reshape(-1, 1)
        if self.has_continuous_action_space:
            action_mean = self.actor(state_subgoal)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            # if self.state_dim == 1:
            #     state = state.reshape(-1,self.state_dim)
            action_probs = self.actor(state_subgoal)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state_subgoal)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, config):
        print(config)
        self.has_continuous_action_space = config['has_continuous_action_space']

        self.action_std = config['action_std_init'] # default 0.6

        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.K_epochs = config['K_epochs']

        self.buffer = RolloutBuffer()

        self.action_dim = config['action_dim']
        self.subgoal_dim = config['subgoal_dim']
        self.state_dim = config['state_dim']

        self.policy_ = ActorCritic(self.state_dim, self.subgoal_dim, self.action_dim, self.has_continuous_action_space,
                                   self.action_std).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_.actor.parameters(), 'lr': config['actor_lr']},
            {'params': self.policy_.critic.parameters(), 'lr': config['critic_lr']}
        ])

        self.policy_old = ActorCritic(self.state_dim, self.subgoal_dim, self.action_dim, self.has_continuous_action_space,
                                       self.action_std).to(device)
        self.policy_old.load_state_dict(self.policy_.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy_.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def policy(self, state, subgoal):
        if self.state_dim == 1:
            state = [state]
        if self.subgoal_dim == 1:
            subgoal = [subgoal]
        state = np.concatenate([state, subgoal])
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # state, hn = self.lstm.forward(state)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device) # Tensor from list is slow, but from np.array doesnt work
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def learn(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def record(self, tuple):
        ppo.buffer.rewards.append(tuple[0])
        ppo.buffer.is_terminals.append(tuple[1])


if __name__ == '__main__':
    env = frozen_lake.FrozenLakeEnv(is_slippery=False)
    #env = gym.make('CartPole-v1')
    config = {
        'actor_lr': 0.0003,
        'critic_lr': 0.0005,
        'action_dim': 4,
        'state_dim': 2,
        'subgoal_dim': 2,
        "gamma": 0.99,
        "eps_clip": 0.2,
        'K_epochs': 10,
        'has_continuous_action_space': False,
        'action_std_init':0.6,
    }
    ppo = PPO(config)
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    total_episodes = 200
    subgoal = 1
    # Takes about 4 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            if ep > 2990:
                env.render()
            action = ppo.policy(prev_state, np.random.randint(0,2, 2))
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
            ppo.record((reward, done))
            episodic_reward += reward

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ppo.learn()
        ep_reward_list.append(episodic_reward)
        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    prev_state = env.reset()
    env.render()
    episodic_reward = 0
    for _ in range(100):
        action = ppo.policy(prev_state,np.random.randint(0,2, 2))
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)
        env.render()
        ppo.record((reward, done))
        episodic_reward += reward

        # End this episode when `done` is True
        if done:
            break

        prev_state = state
    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()



