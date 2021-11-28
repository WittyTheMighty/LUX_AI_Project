import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential
import gym
import scipy.signal
import time
import datetime as dt
import tensorflow_probability as tfp


#Todo: Verify if the policy neural network are correctly implemented.
class ActorCriticPolicy(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.lstm1 = layers.LSTM(1, input_shape=(1, num_actions))
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.value = keras.layers.Dense(1)
        self.policy_logits = keras.layers.Dense(num_actions)

    def call(self, inputs):
        x =self.LSTM(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

class CompositeActorCritic:

    def __init__(self, env,learning_rate,high_env_buffer_size,low_env_buffer_size,n_episode,batch_size):

        self.n_episode = n_episode
        #low_env_buffer
        self.env = env
        self.low_env_buffer =[]
        #DDPG buffer
        self.high_level_agent = DDPG(#Todo: Insert param)

        self.low_level_agent = PPO(#Todo: Create functionnal PPO)


    #Todo: implement the algorithm from the paper
    def composite_hierarchical_actor_critic(self):

        for episode n_episode:

            #Env reset return the initial environment
            s_l = s_h = self.env.reset()

            sub_goal = self.high_level_agent.policy(s_l)
            #For T attemps do:
            for T in range(10):
                #Concatenation sub-goal vectors need to represent thing we can control like:
                # -Ressources
                # -Build house
                # -Collect research point
                # The neural network will take the concatenation between both vectors
                action = self.PPO.low_level_agent.policy(tf.concat([s_l, sub_goal], axis=1)) #+ Todo: Add noise functon
                env_reward,next_State,done = env.step(action)
                #Give reward if agent reaches state
                internal_reward = self.DDPG.compute_internal_reward(next_State)

                #Reward to setup in luxAi Agent policy
                reward = reward + env_reward
                #
                self.low_env_buffer.append((state,sub_goal,next_state,action,internal_reward))
                s_l = next_State
                if self.DDPG.goal_is_done(state)




        pass
    # e
    # def create_nn(self, features, name=None):
    #
    #     if name is None:
    #         name = self.actor_name
    #
    #     with tf.variable_scope(name + '_fc_1'):
    #         fc1 = layer(features, 64)
    #     with tf.variable_scope(name + '_fc_2'):
    #         fc2 = layer(fc1, 64)
    #     with tf.variable_scope(name + '_fc_3'):
    #         fc3 = layer(fc2, 64)
    #     with tf.variable_scope(name + '_fc_4'):
    #         fc4 = layer(fc3, self.action_space_size, is_output=True)
    #
    #     output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset
    #
    #     return output










