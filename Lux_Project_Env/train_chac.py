import argparse
import glob
import os
import sys
import random

from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
from util.chac import CompositeActorCritic

from util.agent_policy import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.constants import LuxMatchConfigs_Default


# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


def get_command_line_arguments():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Training script for Lux RL agent.')
    parser.add_argument('--id', help='Identifier of this run', type=str, default=str(random.randint(0, 10000)))
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--gamma', help='Gamma', type=float, default=0.995)
    parser.add_argument('--gae_lambda', help='GAE Lambda', type=float, default=0.95)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=2048)  # 64
    parser.add_argument('--step_count', help='Total number of steps to train', type=int, default=100000000)
    parser.add_argument('--n_steps', help='Number of experiences to gather before each learning period', type=int,
                        default=2048)
    parser.add_argument('--path', help='Path to a checkpoint to load to resume training', type=str, default=None)
    parser.add_argument('--n_envs', help='Number of parallel environments to use in training', type=int, default=1)
    parser.add_argument('--opponent', help='Select a model to load as an opponent for the training.', type=str,
                        default=None)
    args = parser.parse_args()

    return args


def train(args):
    """
    The main training loop
    :param args: (ArgumentParser) The command line arguments
    """
    print(args)

    # Run a training job
    configs = LuxMatchConfigs_Default

    opponent = None
    if args.opponent is not None:
        try:
            opp_model = PPO.load(args.opponent)
            opponent = AgentPolicy(mode='inference', model=opp_model)
        except:
            print('Could not load given opponent model, uses default instead...')
            opponent = Agent()
    else:
        opponent = Agent()
    # # Create a default opponent agent
    # opponent = Agent()

    # Create a RL agent in training mode
    player = AgentPolicy(mode="train")

    # Train the model
    env_eval = None
    if args.n_envs == 1:
        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)
    else:
        env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agent=opponent), i) for i in range(args.n_envs)])

    run_id = args.id
    print("Run id %s" % run_id)

    model_configs = {
        'batch_size' : 16,
        'n_episodes' : 15000,
        'attempts' : 32,
        'step_limit' : 16
    }
    model_configs["PPO"] = {
        'actor_lr': 0.0003,
        'critic_lr': 0.0005,
        'action_dim': len(player.actions_units),
        'state_dim': player.observation_shape[0],
        'subgoal_dim': 5,
        "gamma": 0.99,
        "eps_clip": 0.2,
        'K_epochs': 10,
        'has_continuous_action_space': False,
        'action_std_init':0.6
    }
    model_configs["DQN"] = {
        'state_dim': player.observation_shape[0],
        'subgoal_dim': 5,  # 5 for current agent_policy
        'lr': 5e-3,         # higher lr seems better
        'sync_freq': 5,
        'exp_replay_size': 256,
        'batch_size': 16,
        'epsilon': 1,
        'epsilon_disc': (1/5000), # around n_episodes / 2 or /3
        'K': 1,
        'gamma': 0.95
    }
    if args.path:
        # by default previous model params are used (lr, batch size, gamma...)
        model = CompositeActorCritic(env, model_configs)
        model.load_checkpoint(args.path)

        # Update the learning rate
        # model.lr_schedule = get_schedule_fn(args.learning_rate)

        # TODO: Update other training parameters
    else:
        model = CompositeActorCritic(env, model_configs)

    callbacks = []

    # Save a checkpoint and 5 match replay files every 100K steps
    player_replay = AgentPolicy(mode="inference", model=model)
    callbacks.append(
        SaveReplayAndModelCallback(
            save_freq=1000000,  ## ADDED x10
            save_path='./models/',
            name_prefix=f'model{run_id}',
            replay_env=LuxEnvironment(
                configs=configs,
                learning_agent=player_replay,
                opponent_agent=Agent()
            ),
            replay_num_episodes=5
        )
    )

    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    if args.n_envs > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        env_eval = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                          learning_agent=AgentPolicy(mode="train"),
                                                          opponent_agent=opponent), i) for i in range(4)])

        callbacks.append(
            EvalCallback(env_eval, best_model_save_path=f'./logs_{run_id}/',
                         log_path=f'./logs_{run_id}/',
                         eval_freq=args.n_steps * 2,  # Run it every 2 training iterations
                         n_eval_episodes=30,  # Run 30 games
                         deterministic=False, render=False)
        )

    print("Training model...")
    # model.learn(total_timesteps=args.step_count,
    #             callback=callbacks)
    model.train_CHAC()
    if not os.path.exists(f'models/rl_model_{run_id}_{args.step_count}_steps'):
        # model.save(path=f'models/rl_model_{run_id}_{args.step_count}_steps.zip')
        model.save_checkpoint(f'models/rl_model_{run_id}_{args.step_count}_steps')
    print("Done training model.")

    # Inference the model
    print("Inference model policy with rendering...")
    saves = glob.glob(f'models/rl_model_{run_id}_*_steps')
    latest_save = sorted(saves, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
    model.load_checkpoint(path=latest_save)
    obs = env.reset()
    for i in range(600):
        action_code, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action_code)
        if i % 5 == 0:
            print("Turn %i" % i)
            env.render()

        if done:
            print("Episode done, resetting.")
            obs = env.reset()
    print("Done")

    '''
    # Learn with self-play against the learned model as an opponent now
    print("Training model with self-play against last version of model...")
    player = AgentPolicy(mode="train")
    opponent = AgentPolicy(mode="inference", model=model)
    env = LuxEnvironment(configs, player, opponent)
    model = PPO("MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./lux_tensorboard/",
        learning_rate = 0.0003,
        gamma=0.999,
        gae_lambda = 0.95
    )

    model.learn(total_timesteps=2000)
    env.close()
    print("Done")
    '''


if __name__ == "__main__":
    if sys.version_info < (3, 7) or sys.version_info >= (3, 8):
        os.system("")


        class style():
            YELLOW = '\033[93m'


        version = str(sys.version_info.major) + "." + str(sys.version_info.minor)
        message = f'/!\ Warning, python{version} detected, you will need to use python3.7 to submit to kaggle.'
        message = style.YELLOW + message
        print(message)

    # Get the command line arguments
    local_args = get_command_line_arguments()

    # Train the model
    train(local_args)
