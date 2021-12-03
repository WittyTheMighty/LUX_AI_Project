import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn import functional as F

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


# Source: https://github.com/MoritzTaylor/ddpg-pytorch/blob/master/utils/nets.py
def fan_in_uniform_init(tensor, fan_in=None):
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG:
    def __init__(self, config):
        self.state_dim = config['state_dim']
        print("Size of State Space ->  {}".format(self.state_dim))
        self.state_n = config['state_n']

        self.subgoal_dim = config['subgoal_dim']
        print("Size of Subgoal Space ->  {}".format(self.subgoal_dim))
        self.subgoal_n = config['subgoal_n']

        self.std_dev = config['std_dev']  # 0.2
        self.ou_noise = OUActionNoise(self, mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.eval()
        self.target_critic.eval()

        # Making the weights equal initially
        hard_update(self.target_actor, self.actor_model)
        hard_update(self.target_critic, self.critic_model)

        # Learning rate for actor-critic models
        self.critic_lr = config['critic_lr']  # 0.002
        self.actor_lr = config['actor_lr']  # 0.001

        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)

        # Discount factor for future rewards
        self.gamma = config['gamma']
        # Used to update target networks
        self.tau = config['tau']

        self.buffer = BufferH(self, 50000, 64)

    def get_actor(self):
        return Actor(self.state_dim, self.subgoal_n)

    def get_critic(self):
        return Critic(self.state_dim, self.subgoal_n)

    # %%
    def policy(self, state):
        if self.state_dim == 1:
            state = [state]
        state = torch.FloatTensor(state).unsqueeze(dim=0)
        self.actor_model.eval()
        probs = self.actor_model(state).squeeze()
        self.actor_model.train()

        # noise = torch.from_numpy(self.ou_noise())
        # Adding noise to action
        # probs = probs + noise
        # We make sure action is within bounds
        # legal_action = probs.clamp(self.lower_bound, self.upper_bound)
        dist = Categorical(probs)
        subgoal = dist.sample()


        return np.array([subgoal.detach().numpy()])[0], probs.detach().numpy()

    def record(self, state, subgoal_probs, reward, prev_state):
        self.buffer.record((prev_state, subgoal_probs, reward, state))

    def learn(self):
        self.buffer.learn()
        self.update_target(self.tau)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    def update_target(self, tau):
        soft_update(self.target_critic, self.critic_model, tau)
        soft_update(self.target_actor, self.actor_model, tau)

class OUActionNoise:
    def __init__(self, ddpg: DDPG, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        self.ddpg = ddpg

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class BufferH:
    def __init__(self, ddpg: DDPG, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.ddpg = ddpg

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, self.ddpg.state_dim))
        self.subgoal_prob_buffer = np.zeros((self.buffer_capacity, self.ddpg.subgoal_n))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.ddpg.state_dim))

    def clear(self):
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, self.ddpg.state_dim))
        self.subgoal_prob_buffer = np.zeros((self.buffer_capacity, self.ddpg.subgoal_n))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.ddpg.state_dim))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.subgoal_prob_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = torch.from_numpy(self.state_buffer[batch_indices]).type(torch.FloatTensor)
        subgoal_prob_batch = torch.from_numpy(self.subgoal_prob_buffer[batch_indices]).type(torch.FloatTensor)
        reward_batch = torch.from_numpy(self.reward_buffer[batch_indices]).type(torch.FloatTensor)
        next_state_batch = torch.from_numpy(self.next_state_buffer[batch_indices]).type(torch.FloatTensor)

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        target_subgoal_probs = self.ddpg.target_actor(next_state_batch).type(torch.FloatTensor)

        y = reward_batch + self.ddpg.gamma * self.ddpg.target_critic([next_state_batch, target_subgoal_probs.detach()])

        self.ddpg.critic_optimizer.zero_grad()
        critic_value = self.ddpg.critic_model([state_batch, subgoal_prob_batch])
        critic_loss = -F.cross_entropy(critic_value, y.detach())
        critic_loss.backward()
        self.ddpg.critic_optimizer.step()

        self.ddpg.actor_optimizer.zero_grad()
        subgoals_probs = self.ddpg.actor_model(state_batch).type(torch.FloatTensor)
        critic_value = self.ddpg.critic_model([state_batch, subgoal_probs])
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -critic_value.mean()
        actor_loss.backward()
        self.ddpg.actor_optimizer.step()

    def __len__(self):
        return self.buffer_counter


# %%
class Actor(nn.Module):
    def __init__(self, state, subgoal):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state, 512),
            nn.ReLU(),
            # Layer Norm is used instead of Batch Norm,
            # as Batch Norm does not work in the same way as in Keras
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, subgoal),
            nn.Softmax(dim=-1)

        )

        self.model[-2].weight.data.uniform_(-0.003, 0.003)
        self.model[-2].bias.data.uniform_(-0.003, 0.003)

    def forward(self, inputs):
        return self.model(inputs)  # * upper_bound


class Critic(nn.Module):
    def __init__(self, state, subgoal):
        super().__init__()

        self.state_model = nn.Sequential(
            nn.Linear(state, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.action_model = nn.Sequential(
            nn.Linear(subgoal, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.out_model = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        model_input = self.state_model(inputs[0])
        action_input = self.action_model(inputs[1])

        return self.out_model(torch.cat((model_input, action_input), dim=1))


if __name__ == '__main__':
    env = frozen_lake.FrozenLakeEnv(is_slippery=False)
    config = {
        'state_dim': 1,  # env.observation_space.shape[0],
        'state_n': 16,
        'subgoal_dim': 1,  # env.action_space.shape[0],
        'subgoal_n': 4,
        'std_dev': 0.2,
        'critic_lr': 0.0001,
        'actor_lr': 0.0001,
        'gamma': 0.99,
        'tau': 0.005,
    }
    ddpg = DDPG(config)
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    total_episodes = 2000

    # Takes about 4 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()
            subgoal, probs = ddpg.policy(prev_state)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(subgoal)
            # print(state, reward,done)
            ddpg.record(state, probs, reward, prev_state)
            episodic_reward += reward

            # End this episode when `done` is True
            if done:
                break

            prev_state = state
        ddpg.learn()
        ddpg.buffer.clear()
        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
