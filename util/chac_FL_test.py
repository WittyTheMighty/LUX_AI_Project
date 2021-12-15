from Lux_Project_Env import frozen_lake
from chac_frozen_lake import SimpleCHAC

if __name__ == '__main__':
    env = frozen_lake.FrozenLakeEnv(map_name='5x5_easy', is_slippery=False)

    model_configs = {'batch_size': 64, 'n_episodes': 3000, 'attempts': 64, 'step_limit': 32, "PPO": {
        'actor_lr': 0.0001,
        'critic_lr': 0.0002,
        'action_dim': 4,
        'state_dim': 1,
        'subgoal_dim': 1,
        "gamma": 0.99,
        "eps_clip": 0.2,
        'K_epochs': 10,
        'has_continuous_action_space': False,
        'action_std_init': 0.6
    }, "DQN": {
        'state_dim': 1,
        'subgoal_dim': 16,  # 5 for current agent_policy
        'lr': 1e-3,  # higher lr seems better
        'sync_freq': 2,
        'exp_replay_size': 256,
        'batch_size': 16,
        'epsilon': 1,
        'epsilon_disc': (1 / 2400),  # around n_episodes / 2 or /3
        'K': 4,
        'gamma': 0.95
    }}

    model = SimpleCHAC(env, model_configs)

    model.train_CHAC()
    model.save_checkpoint('./save/model_save')

    # model.load_checkpoint('./save/model_save')
    obs = env.reset()
    r = 0
    for i in range(600):
        action_code = model.predict(obs)
        obs, rewards, done, info = env.step(action_code)
        r += rewards
        print('obs', obs, 'reward', r, 'done', done)
        # if i % 5 == 0:
        #     print("Turn %i" % i)
        env.render()

        if done:
            obs = env.reset()
            r = 0
    print("Done")
