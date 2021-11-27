import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
import gym
import scipy.signal
import time
import datetime as dt
import tensorflow_probability as tfp


class PPO():

    def __init__(self, config):
        self.config = config
        # Make gym
        self.env = gym.make(config['env'])
        # general
        self.n_observation_space = self.env.observation_space.shape[0]
        self.n_action_space = self.env.action_space.n
        # Algorithm specific variable:

        self.critic_loss_weight = config["critic_loss_weigth"]
        self.entropy_loss_weight = config["entropy_loss_weigth"]
        self.entropy_discount_weight = config["entropy_loss_weigth"]
        self.gamma = config["gamma"]
        self.clip_value = config["entropy_loss_weigth"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config['batch_size']
        self.num_train_epochs = config["num_train_epochs"]
        self.entropy_discount_weight = config["entropy_discount_weight"]

        # Policy model :
        self.model = Model(int(self.n_action_space))
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

    # loss calculations
    def critic_loss(self, discounted_rewards, value_est):
        return tf.cast(
            tf.reduce_mean(keras.losses.mean_squared_error(discounted_rewards, value_est)) * self.critic_loss_weight,
            tf.float32)

    def entropy_loss(self, policy_logits, ent_discount_val):
        probs = tf.nn.softmax(policy_logits)
        entropy_loss = -tf.reduce_mean(keras.losses.categorical_crossentropy(probs, probs))
        return entropy_loss * ent_discount_val

    def actor_loss(self, advantages, old_probs, action_inds, policy_logits):
        probs = tf.nn.softmax(policy_logits)
        new_probs = tf.gather_nd(probs, action_inds)
        ratio = new_probs / old_probs
        policy_loss = -tf.reduce_mean(tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * advantages
        ))
        return policy_loss

    def train_model(self, action_inds, old_probs, states, advantages, discounted_rewards, ent_discount_val):
        with tf.GradientTape() as tape:
            values, policy_logits = self.model.call(tf.stack(states))
            act_loss = self.actor_loss(advantages, old_probs, action_inds, policy_logits)
            ent_loss = self.entropy_loss(policy_logits, ent_discount_val)
            c_loss = self.critic_loss(discounted_rewards, values)
            tot_loss = act_loss + ent_loss + c_loss
        grads = tape.gradient(tot_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tot_loss, c_loss, act_loss, ent_loss

    def get_advantages(self, rewards, dones, values, next_value):
        discounted_rewards = np.array(rewards + [next_value[0]])
        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + self.gamma * discounted_rewards[t + 1] * (1 - dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - np.stack(values)[:, 0]
        # standardise advantages
        advantages -= np.mean(advantages)
        advantages /= (np.std(advantages) + 1e-10)
        # standardise rewards too
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
        return discounted_rewards, advantages

    def policy_gradient_descent(self):
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_writer = tf.summary.create_file_writer(f"/PPO-CartPole_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
        num_steps = 10000000
        episode_reward_sum = 0
        state = self.env.reset()
        episode = 1
        total_loss = None
        ent_discount_val = self.entropy_discount_weight
        for step in range(num_steps):
            rewards = []
            actions = []
            values = []
            states = []
            dones = []
            probs = []
            for _ in range(self.batch_size):
                _, policy_logits = self.model(state.reshape(1, -1))
                action, value = self.model.action_value(state.reshape(1, -1))
                new_state, reward, done, _ = self.env.step(action.numpy()[0])
                actions.append(action)
                values.append(value[0])
                states.append(state)
                dones.append(done)
                probs.append(policy_logits)
                episode_reward_sum += reward
                state = new_state
                if done:
                    rewards.append(0.0)

                    state = self.env.reset()
                    if total_loss is not None:
                        print(f"Episode: {episode}, latest episode reward: {episode_reward_sum}, "
                              f"total loss: {np.mean(total_loss)}, critic loss: {np.mean(c_loss)}, "
                              f"actor loss: {np.mean(act_loss)}, entropy loss {np.mean(ent_loss)}")
                    with train_writer.as_default():
                        tf.summary.scalar('rewards', episode_reward_sum, episode)
                    episode_reward_sum = 0
                    episode += 1
                else:
                    rewards.append(reward)
            _, next_value = self.model.action_value(state.reshape(1, -1))
            discounted_rewards, advantages = self.get_advantages(rewards, dones, values, next_value[0])
            actions = tf.squeeze(tf.stack(actions))
            probs = tf.nn.softmax(tf.squeeze(tf.stack(probs)))
            action_inds = tf.stack([tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1)
            total_loss = np.zeros((self.num_train_epochs))
            act_loss = np.zeros((self.num_train_epochs))
            c_loss = np.zeros(((self.num_train_epochs)))
            ent_loss = np.zeros((self.num_train_epochs))
            for epoch in range(self.num_train_epochs):
                loss_tuple = self.train_model(action_inds, tf.gather_nd(probs, action_inds), states, advantages,
                                              discounted_rewards, ent_discount_val)
                total_loss[epoch] = loss_tuple[0]
                c_loss[epoch] = loss_tuple[1]
                act_loss[epoch] = loss_tuple[2]
                ent_loss[epoch] = loss_tuple[3]
            ent_discount_val *= self.entropy_discount_weight
            with train_writer.as_default():
                tf.summary.scalar('tot_loss', np.mean(total_loss), step)
                tf.summary.scalar('critic_loss', np.mean(c_loss), step)
                tf.summary.scalar('actor_loss', np.mean(act_loss), step)
                tf.summary.scalar('entropy_loss', np.mean(ent_loss), step)

class Model(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.dense1 = layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.value = layers.Dense(1)
        self.policy_logits = layers.Dense(num_actions)


    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, state):
        value, logits = self.predict_on_batch(state)
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value